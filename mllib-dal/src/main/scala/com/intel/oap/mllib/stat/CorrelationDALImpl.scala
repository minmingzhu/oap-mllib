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

import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseMatrix, Matrix, Vector}
import org.apache.spark.rdd.RDD

import java.time.Instant

class CorrelationDALImpl(
                          val executorNum: Int,
                          val executorCores: Int)
  extends Serializable with Logging {

  def computeCorrelationMatrix(data: RDD[Vector]): Matrix = {
    val sparkContext = data.sparkContext
    val metrics_name = "Correlation_" + executorNum
    val corTimer = new Utils.AlgoTimeMetrics(metrics_name, sparkContext)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val storePath = sparkContext.getConf.get("spark.oap.mllib.kvsStorePath") + "/" + Instant.now()
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    corTimer.record("Preprocessing")

    val coalescedTables = if (useDevice == "GPU") {
      OneDAL.coalesceVectorsToFloatHomogenTables(data, executorNum,
        computeDevice)
    } else {
      OneDAL.coalesceVectorsToNumericTables(data, executorNum)
    }
    corTimer.record("Data Convertion")

    val kvsIPPort = getOneCCLIPPort(coalescedTables)
    val training_breakdown_name = "Correlation_training_breakdown_" + executorNum;

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

    if (useDevice == "CPU") {
        coalescedTables.mapPartitionsWithIndex { (rank, table) =>
          OneCCL.init(executorNum, rank, kvsIPPort, training_breakdown_name, storePath)
          Iterator.empty
        }.count()
    }
    corTimer.record("OneCCL Init")

    val results = coalescedTables.mapPartitionsWithIndex { (rank, iter) =>
      val (tableArr : Long, rows : Long, columns : Long) = if (useDevice == "GPU") {
        val parts = iter.next().toString.split("_")
        (parts(0).toLong, parts(1).toLong, parts(2).toLong)
      } else {
        (iter.next().toString.toLong, 0L, 0L)
      }
      logInfo(s"tableArr $tableArr, rows $rows, columns $columns")

      val computeStartTime = System.nanoTime()

      val result = new CorrelationResult()
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      var rCorrelation = 0L
      rCorrelation = cCorrelationTrainDAL(
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

      logInfo(s"CorrelationDAL compute took ${durationCompute} secs")

      val ret = if (rank == 0) {
        val convResultStartTime = System.nanoTime()
        val correlationNumericTable = if (useDevice == "GPU") {
          assert(rCorrelation != 0)
          OneDAL.homogenTableToMatrix(OneDAL.makeHomogenTable(rCorrelation),
            Common.ComputeDevice.HOST)
        } else {
          OneDAL.numericTableToMatrix(OneDAL.makeNumericTable(rCorrelation))
        }
        logInfo(s"correlationNumericTable result ${correlationNumericTable.toArray(0).toString}")
        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"CorrelationDAL result conversion took ${durationCovResult} secs")
        logInfo(s"correlationNumericTable result ${correlationNumericTable.toArray(0).toString}")
        Iterator(correlationNumericTable)
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
    logInfo(s"CorrelationDAL compute end")
    corTimer.record("Training")
    corTimer.print()

    val correlationMatrix = results(0)

    correlationMatrix
  }

  @native private[mllib] def cCorrelationTrainDAL(rank: Int,
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
                                           result: CorrelationResult): Long

  @native def cCorrelationSampleTrainDAL(rank: Int,
                                                       executorNum: Int,
                                                       ip_port: String): Long
}
