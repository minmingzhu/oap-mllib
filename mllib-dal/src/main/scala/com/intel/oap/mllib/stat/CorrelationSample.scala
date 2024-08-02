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
import org.apache.spark.TaskContext
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, split}
import scopt.OptionParser

import java.util.logging.{Level, Logger}


object CorrelationSample {
  LibLoader.loadLibraries()
  private val logger = Logger.getLogger("util.OneDAL")
  private val logLevel = Level.INFO

  def main(args: Array[String]): Unit = {
     val rep_num = args(0).toInt
     val spark = SparkSession
      .builder
      .getOrCreate()

    import spark.implicits._

    val data = spark.sparkContext.parallelize( 1 to rep_num ).repartition(rep_num)
    data.count()
    logger.info(s"getNumPartitions ${data.getNumPartitions}")

    val executorNum = Utils.sparkExecutorNum(data.sparkContext)
    val executorCores = Utils.sparkExecutorCores()
    logger.info(s"executorNum ${executorNum}")
    logger.info(s"executorCores ${executorCores}")
    data.mapPartitionsWithIndex { (rank, iter) =>
      logger.info(s"set ZE_AFFINITY_MASK")
      val resources = TaskContext.get().resources()
      val gpuIndices = resources("gpu").addresses.map(_.toInt)
      logger.info(s"set ZE_AFFINITY_MASK rank is $rank.")
      logger.info(s"gpuIndices is ${gpuIndices.mkString(", ")}.")
      OneCCL.setExecutorEnv("ZE_AFFINITY_MASK", gpuIndices(0).toString())
      Iterator.empty
    }.count()
    val cor = new CorrelationDALImpl(executorNum, executorCores)
    val kvsIPPort = getOneCCLIPPort(data)
    data.mapPartitionsWithIndex { (rank, iter) =>
      logger.info(s"run cCorrelationSampleTrainDAL")
      cor.cCorrelationSampleTrainDAL(rank, executorNum, kvsIPPort)
      Iterator.empty
    }.collect()

    logger.info(s"spark.stop()")
    spark.stop()
  }
}
