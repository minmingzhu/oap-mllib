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

import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{LibLoader, OneCCL, OneDAL, Utils}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, split}
import scopt.OptionParser

import java.util.logging.{Level, Logger}


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

    val data = spark.sparkContext.parallelize( 1 to 24 ).repartition(24)
    logger.info(s"getNumPartitions ${data.getNumPartitions}")

    val useDevice = spark.conf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val executorNum = Utils.sparkExecutorNum(data.sparkContext)
    val executorCores = Utils.sparkExecutorCores()
    logger.info(s"executorNum ${executorNum}")
    logger.info(s"executorCores ${executorCores}")

    val kvsIPPort = getOneCCLIPPort(data)
    data.mapPartitionsWithIndex { (rank, iter) =>
      new CorrelationDALImpl(executorNum, executorCores)
        .cCorrelationSampleTrainDAL(rank, executorNum, kvsIPPort)
      Iterator.empty
    }.collect()
    spark.stop()
  }
}
