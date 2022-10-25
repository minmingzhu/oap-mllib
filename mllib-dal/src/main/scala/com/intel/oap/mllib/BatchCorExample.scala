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
import com.intel.oneapi.dal.table.{Common, HomogenTable}

import scala.io.Source


object BatchCorExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CorrelationExample")
      .getOrCreate()
    val computeDevice = Common.ComputeDevice.getDeviceByName("GPU")
    val result = new CorrelationResult()

    val cor = new CorrelationDALImpl(1, 1)
    val input = "/home/xiaochang/opt/ML/data/HiBench/Correlation/Input/50000/" +
      "part-00000-50f53ad5-b45b-463b-b5c3-0f31f794c601-c000.csv"
    val sourceData = readCSV(input)

    val dataTable = new HomogenTable(sourceData.length,
      sourceData(0).length,
      convertArray(sourceData),
      computeDevice)
    cor.cCorrelationTrainDAL(
            0L,
            1,
            computeDevice.ordinal(),
            0,
            "127.0.0.1_3000",
            result
          )
//    val coalescedTables = rdd.repartition(1).mapPartitionsWithIndex {
//      (index: Int, it: Iterator[Vector]) =>
//        val table = makeHomogenTable(it.toArray, computeDevice)
//        Iterator(table.getcObejct())
//    }.cache()
//    val kvsIPPort = getOneCCLIPPort(coalescedTables)
//    val cor = new CorrelationDALImpl(1, 1)
//    coalescedTables.count()
//    println(s"1")
//    val results = coalescedTables.mapPartitionsWithIndex{ (rank, table) =>
//
//      val tableArr = table.next()
//      println(tableArr)
//      val computeStartTime = System.nanoTime()
//      val result = new CorrelationResult()
//      println(s"2")
//      cor.cCorrelationTrainDAL(
//        tableArr,
//        1,
//        computeDevice.ordinal(),
//        0,
//        kvsIPPort,
//        result
//      )
//      val computeEndTime = System.nanoTime()
//
//      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9
//
//      println(s"CorrelationDAL compute took ${durationCompute} secs")
//      Iterator.empty
//    }.collect()
  }

  def readCSV(path: String): Array[Array[Double]] = {
    val bufferedSource = Source.fromFile(path)
    var matrix: Array[Array[Double]] = Array.empty
    for (line <- bufferedSource.getLines) {
      val cols = line.split(",").map(_.trim.toDouble)
      matrix = matrix :+ cols
    }
    bufferedSource.close
    matrix
  }

  def convertArray(arrayVectors: Array[Array[Double]]): Array[Double] = {
    val numCols: Int = arrayVectors.head.size
    val numRows: Int = arrayVectors.size
    val arrayDouble = new Array[Double](numRows * numCols)
    var index = 0
    for( array: Array[Double] <- arrayVectors) {
      for (i <- 0 until array.toArray.length ) {
        arrayDouble(index) = array(i)
        if (index < (numRows * numCols)) {
          index = index + 1
        }
      }
    }
    arrayDouble
  }
}
