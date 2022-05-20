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
#include <iomanip>
#include <iostream>

#ifdef CPU_GPU_PROFILE
#include "DPCPPGPU.h"
#endif
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "com_intel_oap_mllib_clustering_KMeansDALImpl.h"
#include "oneapi/dal/algo/kmeans.hpp"
#include "communicator.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

typedef std::shared_ptr<homogen_table> HomogenTablePtr;

static jlong doKMeansOneAPICompute(JNIEnv *env, jobject obj, jlong pNumTabData,
                                   jlong pNumTabCenters, jint cluster_num,
                                   jdouble tolerance, jint iteration_num,
                                   jint cComputeDevice, jobject resultObj) {
    std::cout << "oneDAL (native): Do KMeans OneAPI Compute " << std::endl;
    homogen_table *htable = ((HomogenTablePtr *)pNumTabData)->get();
    homogen_table *centroids = ((HomogenTablePtr *)pNumTabCenters)->get();
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(cluster_num)
                                 .set_max_iteration_count(iteration_num)
                                 .set_accuracy_threshold(tolerance);

    std::int64_t rank_id = 0;
    kmeans::train_result<> result_train;
    switch (getComputeDevice(cComputeDevice)) {
    case compute_device::host: {
        result_train = train(kmeans_desc, *htable, *centroids);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case compute_device::cpu: {
        sycl::queue *cpu_queue = getQueue(false);
        auto comm =
            preview::spmd::make_communicator<preview::spmd::backend::ccl>(
                *cpu_queue);
        rank_id = comm.get_rank();
        kmeans::train_input local_input{*htable, *centroids};
        result_train = preview::train(comm, kmeans_desc, local_input);
        break;
    }
    case compute_device::gpu: {
        std::cout << "oneDAL (native): compute gpu " << std::endl;
        sycl::queue *gpu_queue = getQueue(true);
        std::cout << "oneDAL (native): get queue " << std::endl;
        auto comm =
            preview::spmd::make_communicator<preview::spmd::backend::ccl>(
                *gpu_queue);
        std::cout << "oneDAL (native): make communicator " << std::endl;
        rank_id = comm.get_rank();
        std::cout << "oneDAL (native): comm.get_rank()  %d \n" << rank_id << std::endl;
        kmeans::train_input local_input{*htable, *centroids};
        std::cout << "oneDAL (native): train input  %d \n" << std::endl;
        result_train = preview::train(comm, kmeans_desc, local_input);
        std::cout << "oneDAL (native): train result  %d \n" << std::endl;
        break;
    }
#endif
    default: {
        return (jlong)0;
    }
    }

    if (rank_id == 0) {
        printf("iteration_num: %d \n ", iteration_num);
        std::cout << "iteration_num: " << iteration_num << std::endl;
        std::cout << "Iteration count: " <<  result_train.get_iteration_count()
                      << std::endl;
        std::cout << "Objective function value: " << result_train.get_objective_function_value()
                      << std::endl;
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID totalCostField = env->GetFieldID(clazz, "totalCost", "D");
        jfieldID iterationNumField =
            env->GetFieldID(clazz, "iterationNum", "I");
        // Set iteration num for result
        env->SetIntField(resultObj, iterationNumField,
                         result_train.get_iteration_count());
        // Set cost for result
        env->SetDoubleField(resultObj, totalCostField,
                            result_train.get_objective_function_value());

        homogen_table *centroidsTable =
            new homogen_table(result_train.get_model().get_centroids());
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
JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_clustering_KMeansDALImpl_cKMeansOneapiComputeWithInitCenters(
    JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabCenters,
    jint cluster_num, jdouble tolerance, jint iteration_num,
    jint cComputeDevice, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels with " << std::endl;
    jlong ret = doKMeansOneAPICompute(env, obj, pNumTabData, pNumTabCenters,
                                      cluster_num, tolerance, iteration_num,
                                      cComputeDevice, resultObj);

    return ret;
}
