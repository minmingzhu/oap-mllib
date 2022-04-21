package org.apache.spark.ml

import com.intel.oap.mllib.OneDAL
import com.intel.oneapi.dal.table.HomogenTable
import org.apache.spark.SparkContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.util.Random

class oneDALSuite extends FunctionsSuite with Logging {

  import testImplicits._

  test("test sparse vector to CSRNumericTable") {
    val data = Seq(
      Vectors.sparse(3, Seq((0, 1.0), (1, 2.0), (2, 3.0))),
      Vectors.sparse(3, Seq((0, 10.0), (1, 20.0), (2, 30.0))),
      Vectors.sparse(3, Seq.empty),
      Vectors.sparse(3, Seq.empty),
      Vectors.sparse(3, Seq((0, 1.0), (1, 2.0))),
      Vectors.sparse(3, Seq((0, 10.0), (2, 20.0))),
    )
    val df = data.map(Tuple1.apply).toDF("features")
    df.show()
    val rowsRDD = df.rdd.map {
      case Row(features: Vector) => features
    }
    val results = rowsRDD.coalesce(1).mapPartitions { it: Iterator[Vector] =>
      val vectors: Array[Vector] = it.toArray
      val numColumns = vectors(0).size
      val CSRNumericTable = {
        OneDAL.vectorsToSparseNumericTable(vectors, numColumns)
      }
      Iterator(CSRNumericTable.getCNumericTable)
    }.collect()
    val csr = OneDAL.makeNumericTable(results(0))
    val resultMatrix = OneDAL.numericTableToMatrix(csr)
    val matrix = Matrices.fromVectors(data)

    assert((resultMatrix.toArray sameElements matrix.toArray) === true)
  }

  test("test rddLabeledPoint to merged HomogenTables") {
    val data : RDD[LabeledPoint] = generateLabeledPointRDD(sc, 10, 2, 3)
    val df = data.toDF("label", "features")
    df.show()
    val homogenRdd = OneDAL.rddLabeledPointToMergedHomogenTables(df, "label", "features",1 )
    val results = homogenRdd.coalesce(1).mapPartitionsWithIndex {
      case (rank: Int, tables: Iterator[(Long, Long)]) =>
        val (featureTabAddr, lableTabAddr) = tables.next()
        val featureTable = new HomogenTable(featureTabAddr)
        val labelTable = new HomogenTable(lableTabAddr)

        val featureData = featureTable.getDoubleData()
        val labelData = labelTable.getDoubleData()

        Iterator(featureData, labelData)
    }.collect()
    val fData = results(0)
    val lData = results(1)
    println(fData)
  }

  def generateLabeledPointRDD(
                           sc: SparkContext,
                           nexamples: Int,
                           nfeatures: Int,
                           eps: Double,
                           nparts: Int = 2,
                           probOne: Double = 0.5): RDD[LabeledPoint] = {
    val data = sc.parallelize(0 until nexamples, nparts).map { idx =>
      val rnd = new Random(42 + idx)

      val y = if (idx % 2 == 0) 0.0 else 1.0
      val x = Array.fill[Double](nfeatures) {
        rnd.nextGaussian() + (y * eps)
      }
      LabeledPoint(y, Vectors.dense(x))
    }
    data
  }
}



