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
#include "GPU.h"
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "OutputHelpers.hpp"
#include "OneCCL.h"
#include "com_intel_oap_mllib_clustering_KMeansDALImpl.h"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

static jlong doKMeansOneAPICompute(
    JNIEnv *env, jlong pNumTabData, jlong pNumTabCenters, jint clusterNum,
    jdouble tolerance, jint iterationNum,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    std::cout << "oneDAL (native): compute start" << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);
    homogen_table centroids =
        *reinterpret_cast<const homogen_table *>(pNumTabCenters);
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(clusterNum)
                                 .set_max_iteration_count(iterationNum)
                                 .set_accuracy_threshold(tolerance);
    kmeans::train_input local_input{htable, centroids};

    auto t1 = std::chrono::high_resolution_clock::now();
    kmeans::train_result result_train =
        preview::train(comm, kmeans_desc, local_input);
    auto t2 = std::chrono::high_resolution_clock::now();                    ;

    std::cout << "spend training times : "
              << std::chrono::duration_cast<std::chrono::milliseconds>
                                (t2 - t1).count() / 1000
                          << " secs" << std::endl;
    if (isRoot) {
        std::cout << "Iteration count: " << result_train.get_iteration_count()
                              << std::endl;
        std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
        t2 = std::chrono::high_resolution_clock::now();

        std::cout << "KMeans (native) spend training times : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>
                                 (t2 - t1).count() / 1000
                          << " secs" << std::endl;
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

        HomogenTablePtr centroidsPtr = std::make_shared<homogen_table>(
            result_train.get_model().get_centroids());
        saveHomogenTablePtrToVector(centroidsPtr);
        return (jlong)centroidsPtr.get();
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
    jint clusterNum, jdouble tolerance, jint iterationNum,
    jint computeDeviceOrdinal, jintArray gpuIdxArray, jobject resultObj) {
    std::cout << "oneDAL (native): use DPC++ kernels " << std::endl;
    ccl::communicator &cclComm = getComm();
    int rankId = cclComm.rank();
    jlong ret = 0L;
    int nGpu = env->GetArrayLength(gpuIdxArray);
    std::cout << "oneDAL (native): use GPU kernels with " << nGpu << " GPU(s)"
              << std::endl;

    jint *gpuIndices = env->GetIntArrayElements(gpuIdxArray, 0);

    int size = cclComm.size();
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);

    auto queue =
        getAssignedGPU(device, cclComm, size, rankId, gpuIndices, nGpu);

    ccl::shared_ptr_class<ccl::kvs> &kvs = getKvs();
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, size, rankId, kvs);
    ret = doKMeansOneAPICompute(env, pNumTabData, pNumTabCenters, clusterNum,
                                tolerance, iterationNum, comm, resultObj);
    env->ReleaseIntArrayElements(gpuIdxArray, gpuIndices, 0);
    return ret;
}
#endif
