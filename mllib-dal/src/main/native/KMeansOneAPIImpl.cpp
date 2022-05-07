/*******************************************************************************
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
 *******************************************************************************/

#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#endif
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "com_intel_oap_mllib_clustering_KMeansDALImpl.h"
#include "oneapi/dal/spmd/ccl/communicator.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

typedef std::shared_ptr<homogen_table> HomogenTablePtr;


static jlong doKMeansOneAPICompute(
  JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabCenters, jint cluster_num,
  jdouble tolerance, jint iteration_num, jint cComputeDevice, jobject resultObj) {
       homogen_table *htable =
          ((HomogenTablePtr *)pNumTabData)->get();
       homogen_table *centroids =
          ((HomogenTablePtr *)pNumTabCenters)->get();
       const auto kmeans_desc = kmeans::descriptor<>()
                                       .set_cluster_count(cluster_num)
                                       .set_max_iteration_count(iteration_num)
                                       .set_accuracy_threshold(tolerance);

       std::int64_t rank_id = 0 ;
       kmeans::train_result<> result_train;
switch(getComputeDevice(cComputeDevice)) {
       case compute_device::host:{
          result_train = train(kmeans_desc, *htable, *centroids);
          break;
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:{
          sycl::queue *cpu_queue = getQueue(false);
          auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(*cpu_queue);
          rank_id = comm.get_rank();
          auto rank_count = comm.get_rank_count();
          auto input_vec = split_table_by_rows<double>(*cpu_queue, *htable, rank_count);
          kmeans::train_input local_input { input_vec[rank_id], *centroids };
          result_train = preview::train(comm, kmeans_desc, local_input);
          break;
       }
       case compute_device::gpu:{
          sycl::queue *gpu_queue = getQueue(true);
          auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(*gpu_queue);
          rank_id = comm.get_rank();
          auto rank_count = comm.get_rank_count();
          auto input_vec = split_table_by_rows<double>(*gpu_queue, *htable, rank_count);
          kmeans::train_input local_input { input_vec[rank_id], *centroids };
          result_train = preview::train(comm, kmeans_desc, local_input);
          break;
       }
#endif
       default: {
             return (jlong)0;
       }
    }

   if(rank_id == 0) {
        printf("iteration_num: %d \n " , iteration_num);
//        std::cout << "iteration_num: " << iteration_num << std::endl;
//        std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
//        std::cout << "Objective function value: " << result_train.get_objective_function_value() << std::endl;
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID totalCostField = env->GetFieldID(clazz, "totalCost", "D");
        jfieldID iterationNumField =
            env->GetFieldID(clazz, "iterationNum", "I");
        // Set iteration num for result
        env->SetIntField(resultObj, iterationNumField, result_train.get_iteration_count());
        // Set cost for result
        env->SetDoubleField(resultObj, totalCostField, result_train.get_objective_function_value());

        homogen_table *centroidsTable = new homogen_table(result_train.get_model().get_centroids());
        HomogenTablePtr *centroidsPtr = new HomogenTablePtr(centroidsTable);
        return (jlong)centroidsPtr;
   } else {
        return (jlong)0;
   }
}



/*
 * Class:     com_intel_oap_mllib_clustering_KMeansDALImpl
 * Method:    cKMeansOneapiComputeWithInitCenters
 * Signature: (JJIDIIILcom/intel/oap/mllib/clustering/KMeansResult;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_clustering_KMeansDALImpl_cKMeansOneapiComputeWithInitCenters
  (JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabCenters, jint cluster_num, jdouble tolerance,
  jint iteration_num, jint cComputeDevice, jobject resultObj) {
      std::cout << "oneDAL (native): use GPU kernels with GPU(s)"
             << std::endl;
      jlong ret = doKMeansOneAPICompute(
            env, obj, pNumTabData, pNumTabCenters, cluster_num, tolerance,
            iteration_num, cComputeDevice, resultObj);

      return ret;
}
