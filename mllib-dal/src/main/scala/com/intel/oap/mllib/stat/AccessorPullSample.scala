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
import com.intel.oap.mllib.stat.CorrelationSample.logger
import com.intel.oap.mllib.{LibLoader, OneDAL, Utils}
import com.intel.oneapi.dal.table.{Common, HomogenTable, RowAccessor}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.sql.{Row, SparkSession}

import java.util.logging.{Level, Logger}
import scala.util.Random


object AccessorPullSample {
  LibLoader.loadLibraries()
  private val logger = Logger.getLogger("util.OneDAL")
  private val logLevel = Level.INFO

  def generateRDD(sc: SparkContext,
                        nexamples: Long,
                        nfeatures: Int,
                        nclasses: Int,
                        seed: Int = 777,
                        eps: Double = 3.0,
                        nparts: Int = 1): RDD[LabeledPoint] = {
    val data = sc.parallelize( 0L to nexamples, nparts).map { idx =>
      val rnd = new Random(seed + idx)
      val label = rnd.nextInt(nclasses).toDouble
      val feature = Array.fill[Double](nfeatures) {
        rnd.nextGaussian + (label * eps)
      }
      LabeledPoint(label, Vectors.dense(feature))
    }
    data
  }

  def main(args: Array[String]): Unit = {
     val spark = SparkSession
      .builder
      .getOrCreate()

    val computeDevice = Common.ComputeDevice.getDeviceByName("GPU")
    val data = generateRDD(spark.sparkContext, 6000, 5000, 10)
    import spark.implicits._
    val df = data.toDF()
    df.show()
    df.count()
    val rdd = df.select("features").rdd.map {
       case Row(v: Vector) => v
    }

    rdd.mapPartitionsWithIndex { (rank, iter) =>
      val table = OneDAL.makeHomogenTable(iter.toArray, computeDevice)
      logger.info(s"rows :  ${table.getRowCount}")
      logger.info(s"columns :  ${table.getColumnCount}")
      val accessor = new RowAccessor(table.getcObejct(), computeDevice)
      val arrayDouble: Array[Double] = accessor.pullDouble(0, 5000)
      logger.info(arrayDouble.length.toString)
      logger.info(arrayDouble.head.toString)
      val matrix = OneDAL.homogenTableToMatrix(OneDAL.makeHomogenTable(table.getcObejct()),
            computeDevice)
      logger.info(s"matrix result ${matrix.toArray(0).toString}")
      Iterator.empty
    }.count()
    spark.stop()
  }
}
