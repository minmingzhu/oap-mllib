/*
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.oap.mllib.clustering

import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util._
import org.apache.spark.mllib.clustering.{KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import java.time.Instant

class KMeansDALImpl(var nClusters: Int,
                    var maxIterations: Int,
                    var tolerance: Double,
                    val distanceMeasure: String,
                    val centers: Array[OldVector],
                    val executorNum: Int,
                    val executorCores: Int
                   ) extends Serializable with Logging {

  def train(data: RDD[Vector]): MLlibKMeansModel = {
    val sparkContext = data.sparkContext
    val metrics_name = "Kmeans_" + executorNum
    val kmeansTimer = new Utils.AlgoTimeMetrics(metrics_name, sparkContext)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val storePath = sparkContext.getConf.get("spark.oap.mllib.kvsStorePath") + "/" + Instant.now()
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    kmeansTimer.record("Preprocessing")

    val coalescedTables = if (useDevice == "GPU") {
      OneDAL.coalesceVectorsToFloatHomogenTables(data, executorNum, computeDevice)
    } else {
      OneDAL.coalesceVectorsToNumericTables(data, executorNum)
    }
    kmeansTimer.record("Data Convertion")

    coalescedTables.mapPartitionsWithIndex { (rank, iter) =>
      logInfo(s"set ZE_AFFINITY_MASK")
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      logInfo(s"set ZE_AFFINITY_MASK rank is $rank.")
      logInfo(s"gpuIndices is ${gpuIndices.mkString(", ")}.")
      OneCCL.setExecutorEnv("ZE_AFFINITY_MASK", gpuIndices(0).toString())
      Iterator.empty
    }.count()

    val kvsIPPort = getOneCCLIPPort(coalescedTables)
    val training_breakdown_name = "Kmeans_training_breakdown_" + executorNum;
    coalescedTables.mapPartitionsWithIndex { (rank, iter) =>
      logInfo(s"OneCCL.init")
      OneCCL.init(executorNum, rank, kvsIPPort, training_breakdown_name, storePath)
      Iterator.empty
    }.count()
    kmeansTimer.record("OneCCL Init")
    logInfo(s"OneCCL init finished")

    val results = coalescedTables.mapPartitionsWithIndex { (rank, iter) =>
      logInfo(s"coalescedTables.mapPartitionsWithIndex")
      var cCentroids = 0L
      val result = new KMeansResult()
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }

      val (tableArr : Long, rows : Long, columns : Long) = if (useDevice == "GPU") {
        val parts = iter.next().toString.split("_")
        (parts(0).toLong, parts(1).toLong, parts(2).toLong)
      } else {
        (iter.next().toString.toLong, 0L, 0L)
      }
      logInfo(s"tableArr $tableArr, rows $rows, columns $columns")
      val initCentroids = if (useDevice == "GPU") {
        OneDAL.makeHomogenTable(centers, computeDevice).getcObejct()
      } else {
        OneDAL.makeNumericTable(centers).getCNumericTable
      }
      logInfo(s"initCentroids HomogenTable")

      cCentroids = cKMeansOneapiComputeWithInitCenters(
        rank,
        tableArr,
        rows,
        columns,
        initCentroids,
        nClusters,
        tolerance,
        maxIterations,
        executorNum,
        executorCores,
        computeDevice.ordinal(),
        gpuIndices,
        training_breakdown_name,
        result
      )
      logInfo(s"convert cCentroids HomogenTable to vector start")
      val ret = if (rank == 0) {
          assert(cCentroids != 0)
          val centerVectors = if (useDevice == "GPU") {
            OneDAL.homogenTableToVectors(OneDAL.makeHomogenTable(cCentroids),
              computeDevice)
          } else {
            OneDAL.numericTableToVectors(OneDAL.makeNumericTable(cCentroids))
          }
          Iterator((centerVectors, result.getTotalCost, result.getIterationNum))
        } else {
          Iterator.empty
        }
      logInfo(s"convert cCentroids HomogenTable to vector end")
      OneCCL.cleanup()
      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)
    kmeansTimer.record("Training")
    kmeansTimer.print()

    val centerVectors = results(0)._1
    val totalCost = results(0)._2
    val iterationNum = results(0)._3

    if (iterationNum == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iterationNum iterations.")
    }

    logInfo(s"The cost is $totalCost.")
    logInfo(s"OneDAL output centroids:\n${centerVectors.mkString("\n")}")

    val parentModel = new MLlibKMeansModel(
      centerVectors.map(OldVectors.fromML(_)),
      distanceMeasure, totalCost, iterationNum)

    parentModel
  }

  @native private[mllib] def cKMeansOneapiComputeWithInitCenters(rank: Int,
                                                       data: Long,
                                                       numRows: Long,
                                                       numCols: Long,
                                                       centers: Long,
                                                       clusterNum: Int,
                                                       tolerance: Double,
                                                       iterationNum: Int,
                                                       executorNum: Int,
                                                       executorCores: Int,
                                                       computeDeviceOrdinal: Int,
                                                       gpuIndices: Array[Int],
                                                       training_breakdown_name: String,
                                                       result: KMeansResult): Long
}
