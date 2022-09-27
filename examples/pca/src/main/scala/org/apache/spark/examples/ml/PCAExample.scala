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

// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, concat_ws, split}
// $example off$
import org.apache.spark.sql.SparkSession

object PCAExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("PCAExample")
      .getOrCreate()

    import spark.implicits._
    // $example on$
    // Loads data.
    var df = spark.read.option("quote", " ").csv(args(0))
    df = df.withColumn("features", split(concat_ws(",",   df.schema.fieldNames.map(c=> col(c)):_*), ",") ).select("features")
    df = df.withColumn("features", col("features").cast("array<double>"))
    df = df.withColumn("features", array_to_vector(col("features")))
    df.cache()
    df.show(false)

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(10)
      .fit(df)

    val result = pca.transform(df).select("pcaFeatures")
    result.show(false)
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
