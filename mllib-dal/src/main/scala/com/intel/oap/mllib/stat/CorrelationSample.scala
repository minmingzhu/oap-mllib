/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.oap.mllib.stat

import com.intel.oap.mllib.OneDAL.logger
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{LibLoader, OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.TaskContext
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.rdd.{ExecutorInProcessCoalescePartitioner, RDD}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, split}
import scopt.OptionParser

import java.util.logging.{Level, Logger}
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.mutable
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.{Duration, DurationInt}
import scala.concurrent.{Await, Future}

object CorrelationSample {
  private val logger = Logger.getLogger("util.OneDAL")
  private val logLevel = Level.INFO
  case class Params(
                     input: String = null,
                     corrType: String = "pearson",
                     minPartitionNum: Int = 1,
                     maxPartitionBytes: String = "1g",
                     device: String = "HOST")

  def main(args: Array[String]): Unit = {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("Correlation") {
      head("Correlation: an example app for computing correlations.")
      opt[String]("corrType")
        .text(s"String specifying the method to use for computing correlation. " +
          s"Supported: `pearson` (default), `spearman`, default: ${defaultParams.corrType}")
        .action((x, c) => c.copy(corrType = x))
      opt[Int]("minPartitionNum")
        .text(s"minPartitionNum, default: 1")
        .action((x, c) => c.copy(minPartitionNum = x))
      opt[String]("maxPartitionBytes")
        .text(s"maxPartitionBytes, default: 1g")
        .action((x, c) => c.copy(maxPartitionBytes = x))
      opt[String]("device")
        .text(s"oneapi device, default: host")
        .action((x, c) => c.copy(device = x))
      arg[String]("<input>")
        .text("input path to labeled examples")
        .required()
        .action((x, c) => c.copy(input = x))
    }
    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"Correlations with $params")
      .config("spark.sql.files.maxPartitionBytes", params.maxPartitionBytes)
      .config("spark.sql.files.minPartitionNum", params.minPartitionNum)
      .config("spark.oap.mllib.device", params.device)
      .getOrCreate()
    logger.info(params.toString)

    import spark.implicits._
    logger.info(s"loading data")
    logger.info(params.input)
//    var data = spark.read.option("quote", " ").csv(params.input).toDF("features")
//    data = data.select(split(col("features"), ",").alias("features"))
//    data = data.withColumn("features", col("features").cast("array<double>"))
//    data = data.withColumn("features", array_to_vector(col("features")))
//    data.printSchema()
//    data.cache()
//    data.count()
//    data.show()
//
//    val rdd = data.select("features").rdd.map {
//      case Row(v: Vector) => v
//    }
//    rdd.getNumPartitions
//
    val data = spark.sparkContext.parallelize( 1 to 24 ).repartition(24)
    logger.info(s"getNumPartitions ${data.getNumPartitions}")

    val useDevice = spark.conf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    val executorNum = Utils.sparkExecutorNum(data.sparkContext)
    val executorCores = Utils.sparkExecutorCores()
    logger.info(s"executorNum ${executorNum}")
    logger.info(s"executorCores ${executorCores}")
//
//    logger.info(s"coalesceVectorsToFloatHomogenTables")

//    val hTables = coalesceVectorsToFloatHomogenTables(rdd, executorNum,
//      computeDevice)
    logger.info(s"hTables")
    val kvsIPPort = getOneCCLIPPort(data)
    val breakdown_name = "Correlation_training_breakdown_" + executorNum
    val result = data.mapPartitionsWithIndex { (rank, iter) =>
      OneCCL.init(executorNum, rank, kvsIPPort, breakdown_name)
//      val (tableArr : Long, rows : Long, columns : Long) = if (useDevice == "GPU") {
//      val parts = iter.next().toString.split("_")
//        (parts(0).toLong, parts(1).toLong, parts(2).toLong)
//      } else {
//        (iter.next().toString.toLong, 0L, 0L)
//      }

      val computeStartTime = System.nanoTime()

      val result = new CorrelationResult()
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      new CorrelationDALImpl(executorNum, executorCores).cCorrelationTrainDAL(
      0,
        0,
        0,
      executorNum,
      executorCores,
      computeDevice.ordinal(),
      gpuIndices,
      breakdown_name,
      result
    )

    val computeEndTime = System.nanoTime()

    val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logger.info(s"CorrelationDAL compute took ${durationCompute} secs")

    val ret = if (rank == 0) {
      val convResultStartTime = System.nanoTime()
      val correlationNumericTable = if (useDevice == "GPU") {
        OneDAL.homogenTableToMatrix(OneDAL.makeHomogenTable(result.getCorrelationNumericTable),
          computeDevice)
      } else {
        OneDAL.numericTableToMatrix(OneDAL.makeNumericTable(result.getCorrelationNumericTable))
      }
      val convResultEndTime = System.nanoTime()

      val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

      logger.info(s"CorrelationDAL result conversion took ${durationCovResult} secs")

      Iterator(correlationNumericTable)
    } else {
      Iterator.empty
    }
    OneCCL.cleanup()
    ret
  }.collect()
    spark.stop()
  }

//  def coalesceVectorsToFloatHomogenTables(data: RDD[Vector], executorNum: Int,
//                                          device: Common.ComputeDevice): RDD[String] = {
//    logger.info(s"coalesceVectorsToFloatHomogenTables")
//
//    val numberCores: Int = data.sparkContext.getConf.getInt("spark.executor.cores", 1)
//    // convert RDD to HomogenTable
//    val coalescedTables = data.mapPartitionsWithIndex { (index: Int, it: Iterator[Vector]) =>
//      val list = it.toList
//      val subRowCount: Int = list.size / numberCores
//      val futureList: ListBuffer[Future[Long]] = new ListBuffer[Future[Long]]()
//      val numRows = list.size
//      val numCols = list(0).toArray.size
//      val size = numRows.toLong * numCols.toLong
//      val targetArrayAddress = OneDAL.cNewFloatArray(size)
//      for ( i <- 0 until  numberCores) {
//        val f = Future {
//          val iter = list.iterator
//          val slice = if (i == numberCores - 1) {
//            iter.slice(subRowCount * i, numRows)
//          } else {
//            iter.slice(subRowCount * i, subRowCount * i + subRowCount)
//          }
//          slice.toArray.zipWithIndex.map { case (vector, index) =>
//            val length = vector.toArray.length
//            OneDAL.cCopyFloatArrayToNative(targetArrayAddress,
//              vector.toArray, subRowCount.toLong * numCols * i + length * index)
//          }
//          targetArrayAddress
//        }
//        futureList += f
//      }
//      val result = Future.sequence(futureList)
//      Await.result(result, Duration.Inf)
//
//      Iterator(targetArrayAddress + "_" + numRows.toLong + "_" + numCols.toLong)
//    }.setName("coalescedTables").cache()
//    coalescedTables.count()
//    coalescedTables
//  }
}
