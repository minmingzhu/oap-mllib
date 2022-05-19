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
import com.intel.oap.mllib.{OneCCL, OneDAL}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util._
import org.apache.spark.mllib.clustering.{KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.rdd.RDD

class KMeansDALImpl(var nClusters: Int,
                    var maxIterations: Int,
                    var tolerance: Double,
                    val distanceMeasure: String,
                    val centers: Array[OldVector],
                    val executorNum: Int,
                    val executorCores: Int
                   ) extends Serializable with Logging {

  def train(data: RDD[Vector]): MLlibKMeansModel = {
    System.out.println("KMeansDALImpl")
    val coalescedTables = OneDAL.rddVectorToMergedTables(data, executorNum)

    val kvsIPPort = getOneCCLIPPort(coalescedTables)

    val sparkContext = data.sparkContext
    val isDPC = sparkContext.getConf.getBoolean("spark.oap.mllib.isDPC", false)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", "GPU")
    val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>
      var cCentroids = 0L
      val result = new KMeansResult()
      val tableArr = table.next()
      if (isDPC) {
        System.out.println("KMeansDALImpl isDPC")
        val computeDevice = if (useDevice.toUpperCase().equals("GPU")) {
          System.out.println("KMeansDALImpl GPU")
          Common.ComputeDevice.GPU
        } else if (useDevice.equals("CPU")) {
          System.out.println("KMeansDALImpl CPU")
          Common.ComputeDevice.CPU
        } else {
          System.out.println("KMeansDALImpl HOST")
          Common.ComputeDevice.HOST
        }
        System.out.println("KMeansDALImpl init start")
        OneCCL.initDpcpp()
        System.out.println("KMeansDALImpl init end")
        val initCentroids = OneDAL.makeHomogenTable(centers, computeDevice)
        cCentroids = cKMeansOneapiComputeWithInitCenters(
          tableArr,
          initCentroids.getcObejct(),
          nClusters,
          tolerance,
          maxIterations,
          computeDevice.ordinal(),
          result
        )
      } else {
        OneCCL.init(executorNum, rank, kvsIPPort)
        val initCentroids = OneDAL.makeNumericTable(centers)
        cCentroids = cKMeansDALComputeWithInitCenters(
          tableArr,
          initCentroids.getCNumericTable,
          nClusters,
          tolerance,
          maxIterations,
          executorNum,
          executorCores,
          result
        )
      }

      val ret = if (OneCCL.isRoot()) {
        assert(cCentroids != 0)
        if (isDPC) {
          val centerVectors = OneDAL.homogenTableToVectors(OneDAL.makeHomogenTable(cCentroids),
            Common.ComputeDevice.GPU)
          Iterator((centerVectors, result.totalCost, result.iterationNum))
        } else {
          val centerVectors = OneDAL.numericTableToVectors(OneDAL.makeNumericTable(cCentroids))
          Iterator((centerVectors, result.totalCost, result.iterationNum))
        }
      } else {
        Iterator.empty
      }

      OneCCL.cleanup()

      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

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

  // Single entry to call KMeans DAL backend with initial centers, output centers
  @native private def cKMeansDALComputeWithInitCenters(data: Long, centers: Long,
                                                       cluster_num: Int,
                                                       tolerance: Double,
                                                       iteration_num: Int,
                                                       executor_num: Int,
                                                       executor_cores: Int,
                                                       result: KMeansResult): Long

  // Single entry to call DPC++ KMeans oneapi backend with initial centers, output centers
  @native private[mllib] def cKMeansOneapiComputeWithInitCenters(data: Long, centers: Long,
                                                       cluster_num: Int,
                                                       tolerance: Double,
                                                       iteration_num: Int,
                                                       computeDevice: Int,
                                                       result: KMeansResult): Long
}
