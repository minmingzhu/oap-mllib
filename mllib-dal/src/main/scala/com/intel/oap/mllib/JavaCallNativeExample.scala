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

import com.intel.oap.mllib.stat.{CorrelationDALImpl, CorrelationResult}
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import scala.io.Source

object JavaCallNativeExample {
  def main(args: Array[String]): Unit = {
    val input = args(0)
    printf(s"load csv file")
    val sourceData = readCSV(input)
    printf(s"create homogentable")
    val dataTable = new HomogenTable(sourceData.length, sourceData(0).length,
      convertArray(sourceData), Common.ComputeDevice.GPU)

    val correlationDAL = new CorrelationDALImpl(1, 1)
    val result = new CorrelationResult()
    printf(s"call native compute")
    correlationDAL.cCorrelationTrainDAL(dataTable.getcObejct(), 1,
        Common.ComputeDevice.GPU.ordinal(), 0, "127.0.0.1_3000", result)

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
