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

package com.intel.oap.mllib

import com.intel.oap._
import com.intel.oap.mllib.OneDAL.makeHomogenTable
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.stat.CorrelationDALImpl
import com.intel.oap.mllib.stat.CorrelationResult
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.functions.{col, split}
import org.apache.spark.sql.{Row, SparkSession}
import com.intel.oneapi.dal.table.Common


object BatchCorExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CorrelationExample")
      .getOrCreate()
    val input = args(0)
    var data = spark.read.option("quote", " ").csv(input).toDF("features")
    data = data.select(split(col("features"), ",").alias("features"))
    data = data.withColumn("features", col("features").cast("array<double>"))
    data = data.withColumn("features", array_to_vector(col("features")))
    data.cache()
    data.show()
    val rdd = data.select("features").rdd.map {
      case Row(v: Vector) => v
    }
    val computeDevice = Common.ComputeDevice.getDeviceByName("GPU")
    val coalescedTables = rdd.repartition(1).mapPartitionsWithIndex {
      (index: Int, it: Iterator[Vector]) =>
        val table = makeHomogenTable(it.toArray, computeDevice)
        Iterator(table.getcObejct())
    }.cache()
    val kvsIPPort = getOneCCLIPPort(coalescedTables)
    val cor = new CorrelationDALImpl(1, 1)
    coalescedTables.count()
    println(s"1")
    val results = coalescedTables.mapPartitionsWithIndex{ (rank, table) =>

      val tableArr = table.next()
      println(tableArr)
      val computeStartTime = System.nanoTime()
      val result = new CorrelationResult()
      println(s"2")
      cor.cCorrelationTrainDAL(
        tableArr,
        1,
        computeDevice.ordinal(),
        0,
        kvsIPPort,
        result
      )
      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      println(s"CorrelationDAL compute took ${durationCompute} secs")
      Iterator.empty
    }.collect()
  }
}
