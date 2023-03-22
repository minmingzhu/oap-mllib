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
package com.intel.oap.mllib.classification

import com.google.common.collect.HashBiMap
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.tree.{InternalNode, LeafNode, Node, Split}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.tree
import org.apache.spark.mllib.tree.model.ImpurityStats

import java.util
import java.util.{ArrayList, Map}
import scala.collection.mutable.HashMap

class RandomForestClassificationModel private[mllib] (
                           val uid: String,
                         val _trees: Array[DecisionTreeClassificationModel],
                         val numFeatures: Int,
                         val numClasses: Int)

//class DecisionTreeClassificationModel private[mllib] (
//                          val uid: String,
//                          val rootNode: Node,
//                          val numFeatures: Int,
//                          val numClasses: Int)
//
//private[mllib] class LearningNode(
//                                  var id: Int,
//                                  var leftChild: Option[LearningNode],
//                                  var rightChild: Option[LearningNode],
//                                  var split: Option[Split],
//                                  var isLeaf: Boolean,
//                                  var stats: ImpurityStats)extends Serializable {
//
//  def toNode: Node = toNode(prune = true)
//
//  /**
//   * Convert this [[LearningNode]] to a regular [[Node]], and recurse on any children.
//   */
//  def toNode(prune: Boolean = true): Node = {
//
//    if (!leftChild.isEmpty || !rightChild.isEmpty) {
//      assert(leftChild.nonEmpty && rightChild.nonEmpty && split.nonEmpty && stats != null,
//        "Unknown error during Decision Tree learning.  Could not convert LearningNode to Node.")
//      (leftChild.get.toNode(prune), rightChild.get.toNode(prune)) match {
//        case (l: LeafNode, r: LeafNode) if prune && l.prediction == r.prediction =>
//          new LeafNode(l.prediction, stats.impurity, stats.impurityCalculator)
//        case (l, r) =>
//          new InternalNode(stats.impurityCalculator.predict, stats.impurity, stats.gain,
//            l, r, split.get, stats.impurityCalculator)
//      }
//    } else {
//      if (stats.valid) {
//        new LeafNode(stats.impurityCalculator.predict, stats.impurity,
//          stats.impurityCalculator)
//      } else {
//        // Here we want to keep same behavior with the old mllib.DecisionTreeModel
//        new LeafNode(stats.impurityCalculator.predict, -1.0, stats.impurityCalculator)
//      }
//    }
//  }
//}

class RandomForestClassifierDALImpl(val uid: String,
                                    val classCount: Int,
                                    val treeCount: Int,
                                    val featurePerNode: Int,
                                    val minObservationsLeafNode: Int,
                                    val minObservationsSplitNode: Int,
                                    val minWeightFractionLeafNode: Double,
                                    val minImpurityDecreaseSplitNode: Double,
                                    val executorNum: Int,
                                    val executorCores: Int,
                                    val bootstrap: Boolean) extends Serializable with Logging {

  def train(labeledPoints: Dataset[_],
            labelCol: String,
            featuresCol: String): (Matrix, Matrix,
                                   util.Map[Integer, util.ArrayList[LearningNode]]) = {
    logInfo(s"RandomForestClassifierDALImpl executorNum : " + executorNum)
    val sparkContext = labeledPoints.rdd.sparkContext
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    val labeledPointsTables = if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
      OneDAL.rddLabeledPointToMergedHomogenTables(labeledPoints,
        labelCol, featuresCol, executorNum, computeDevice)
    } else {
      OneDAL.rddLabeledPointToSparseCSRTables(labeledPoints,
        labelCol, featuresCol, executorNum, computeDevice)
    }
    val kvsIPPort = getOneCCLIPPort(labeledPointsTables)

    val results = labeledPointsTables.mapPartitionsWithIndex {
      (rank: Int, tables: Iterator[(Long, Long)]) =>
      val (featureTabAddr, lableTabAddr) = tables.next()

      OneCCL.initDpcpp()

      val computeStartTime = System.nanoTime()
      val result = new RandomForestResult
      cRFClassifierTrainDAL(
        featureTabAddr,
        lableTabAddr,
        executorNum,
        computeDevice.ordinal(),
        classCount,
        rank,
        treeCount,
        minObservationsLeafNode,
        minObservationsSplitNode,
        minWeightFractionLeafNode,
        minImpurityDecreaseSplitNode,
        bootstrap,
        kvsIPPort,
        result)

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"RandomForestClassifierDAL compute took ${durationCompute} secs")

      val ret = if (rank == 0) {
        val convResultStartTime = System.nanoTime()
        val probabilitiesNumericTable = OneDAL.homogenTableToMatrix(
          OneDAL.makeHomogenTable(result.probabilitiesNumericTable),
          computeDevice)
        val predictionNumericTable = OneDAL.homogenTableToMatrix(
          OneDAL.makeHomogenTable(result.predictionNumericTable),
          computeDevice)
        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"RandomForestClassifierDAL result conversion took ${durationCovResult} secs")

        Iterator((probabilitiesNumericTable, predictionNumericTable, result.treesMap))
      } else {
        Iterator.empty
      }

      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)
    (results(0)._1, results(0)._2, results(0)._3)
  }


  @native private[mllib] def cRFClassifierTrainDAL(featureTabAddr: Long, lableTabAddr: Long,
                                             executorNum: Int,
                                             computeDeviceOrdinal: Int,
                                             classCount: Int,
                                             rankId: Int,
                                             treeCount: Int,
                                             minObservationsLeafNode: Int,
                                             minObservationsSplitNode: Int,
                                             minWeightFractionLeafNode: Double,
                                             minImpurityDecreaseSplitNode: Double,
                                             bootstrap: Boolean,
                                             ipPort: String,
                                             result: RandomForestResult): Unit
}