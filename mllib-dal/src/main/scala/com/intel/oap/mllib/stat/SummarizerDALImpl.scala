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

package com.intel.oap.mllib.stat

import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalDALSummary, MultivariateStatisticalSummary => Summary}
import org.apache.spark.rdd.RDD
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oneapi.dal.table.Common

import java.time.Instant

class SummarizerDALImpl(val executorNum: Int,
                        val executorCores: Int)
  extends Serializable with Logging {

  def computeSummarizerMatrix(data: RDD[Vector]): Summary = {
    val sparkContext = data.sparkContext
    val metrics_name = "Summarizer_" + executorNum
    val sumTimer = new Utils.AlgoTimeMetrics(metrics_name, sparkContext)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val storePath = sparkContext.getConf.get("spark.oap.mllib.kvsStorePath") + "/" + Instant.now()
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    sumTimer.record("Preprocessing")

    val coalescedTables = if (useDevice == "GPU") {
      OneDAL.coalesceVectorsToFloatHomogenTables(data, executorNum,
        computeDevice)
    } else {
      OneDAL.coalesceVectorsToNumericTables(data, executorNum)
    }
    sumTimer.record("Data Convertion")

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

    val kvsIPPort = getOneCCLIPPort(data)
    val training_breakdown_name = "Summarizer_training_breakdown_" + executorNum;

    if (useDevice == "CPU") {
        coalescedTables.mapPartitionsWithIndex { (rank, table) =>
          OneCCL.init(executorNum, rank, kvsIPPort, training_breakdown_name, storePath)
          Iterator.empty
        }.count()
    }
    sumTimer.record("OneCCL Init")

    val results = coalescedTables.mapPartitionsWithIndex { (rank, iter) =>
      val (tableArr : Long, rows : Long, columns : Long) = if (useDevice == "GPU") {
        val parts = iter.next().toString.split("_")
        (parts(0).toLong, parts(1).toLong, parts(2).toLong)
      } else {
        (iter.next().toString.toLong, 0L, 0L)
      }

      val computeStartTime = System.nanoTime()

      val result = new SummarizerResult()
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      cSummarizerTrainDAL(
        rank,
        tableArr,
        rows,
        columns,
        executorNum,
        executorCores,
        computeDevice.ordinal(),
        gpuIndices,
        kvsIPPort,
        training_breakdown_name,
        storePath,
        result
      )

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"SummarizerDAL compute took ${durationCompute} secs")

      val ret = if (rank == 0) {

        val convResultStartTime = System.nanoTime()
        val meanVector = if (useDevice == "GPU") {
          OneDAL.homogenTable1xNToVector(
            OneDAL.makeHomogenTable(result.getMeanNumericTable), Common.ComputeDevice.HOST)
        } else {
          OneDAL.numericTable1xNToVector(
            OneDAL.makeNumericTable(result.getMeanNumericTable))
        }
        val varianceVector = if (useDevice == "GPU") {
          OneDAL.homogenTable1xNToVector(
            OneDAL.makeHomogenTable(result.getVarianceNumericTable), Common.ComputeDevice.HOST)
        } else {
          OneDAL.numericTable1xNToVector(
            OneDAL.makeNumericTable(result.getVarianceNumericTable))
        }
        val maxVector = if (useDevice == "GPU") {
          OneDAL.homogenTable1xNToVector(
            OneDAL.makeHomogenTable(result.getMaximumNumericTable), Common.ComputeDevice.HOST)
        } else {
          OneDAL.numericTable1xNToVector(
            OneDAL.makeNumericTable(result.getMaximumNumericTable))
        }
        val minVector = if (useDevice == "GPU") {
          OneDAL.homogenTable1xNToVector(
            OneDAL.makeHomogenTable(result.getMinimumNumericTable), Common.ComputeDevice.HOST)
        } else {
          OneDAL.numericTable1xNToVector(
            OneDAL.makeNumericTable(result.getMinimumNumericTable))
        }

        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"SummarizerDAL result conversion took ${durationCovResult} secs")

        Iterator((meanVector, varianceVector, maxVector, minVector))
      } else {
        Iterator.empty
      }
      if (useDevice == "CPU") {
         OneCCL.cleanup()
      }
      ret
    }.collect()
    // Make sure there is only one result from rank 0
    assert(results.length == 1)
    sumTimer.record("Training")
    sumTimer.print()



    val meanVector = results(0)._1
    val varianceVector = results(0)._2
    val maxVector = results(0)._3
    val minVector = results(0)._4

    val summary = new MultivariateStatisticalDALSummary(OldVectors.fromML(meanVector),
                                                        OldVectors.fromML(varianceVector),
                                                        OldVectors.fromML(maxVector),
                                                        OldVectors.fromML(minVector))

    summary
  }

  @native private[mllib] def cSummarizerTrainDAL(rank: Int,
                                          data: Long,
                                          numRows: Long,
                                          numCols: Long,
                                          executorNum: Int,
                                          executorCores: Int,
                                          computeDeviceOrdinal: Int,
                                          gpuIndices: Array[Int],
                                          kvsIPPort: String,
                                          training_breakdown_name: String,
                                          storePath: String,
                                          result: SummarizerResult): Long
}
