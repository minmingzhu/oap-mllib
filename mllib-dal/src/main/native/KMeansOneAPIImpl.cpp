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
#include "communicator.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;

typedef std::shared_ptr<homogen_table> HomogenTablePtr;
static jlong doKMeansHostOneAPICompute(JNIEnv *env, jint rankId,
                                       jlong pNumTabData, jlong pNumTabCenters,
                                       jint cluster_num, jdouble tolerance,
                                       jint iteration_num, jint executor_num,
                                       const ccl::string &ipPort,
                                       cl::sycl::queue *, jobject resultObj) {
    std::cout << "oneDAL (native): HOST compute start , rankid %ld " << rankId
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    homogen_table *htable = ((HomogenTablePtr *)pNumTabData)->get();
    homogen_table *centroids = ((HomogenTablePtr *)pNumTabCenters)->get();
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(cluster_num)
                                 .set_max_iteration_count(iteration_num)
                                 .set_accuracy_threshold(tolerance);
    auto result_train = train(kmeans_desc, *htable, *centroids);
    if (isRoot) {
        printf("iteration_num: %d \n ", iteration_num);
        std::cout << "iteration_num: " << iteration_num << std::endl;
        std::cout << "Iteration count: " << result_train.get_iteration_count()
                  << std::endl;
        std::cout << "Objective function value: "
                  << result_train.get_objective_function_value() << std::endl;
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

#ifdef CPU_GPU_PROFILE
static jlong doKMeansCPUOneAPICompute(
    JNIEnv *env, jint rankId, jlong pNumTabData, jlong pNumTabCenters,
    jint cluster_num, jdouble tolerance, jint iteration_num, jint executor_num,
    const ccl::string &ipPort, cl::sycl::queue *cpu_queue, jobject resultObj) {
    std::cout << "oneDAL (native): CPU compute start , rankid %ld " << rankId
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    homogen_table *htable = ((HomogenTablePtr *)pNumTabData)->get();
    homogen_table *centroids = ((HomogenTablePtr *)pNumTabCenters)->get();
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(cluster_num)
                                 .set_max_iteration_count(iteration_num)
                                 .set_accuracy_threshold(tolerance);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        *cpu_queue, executor_num, rankId, ipPort);
    kmeans::train_input local_input{*htable, *centroids};
    auto result_train = preview::train(comm, kmeans_desc, local_input);
    if (isRoot) {
        printf("iteration_num: %d \n ", iteration_num);
        std::cout << "iteration_num: " << iteration_num << std::endl;
        std::cout << "Iteration count: " << result_train.get_iteration_count()
                  << std::endl;
        std::cout << "Objective function value: "
                  << result_train.get_objective_function_value() << std::endl;
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

static jlong doKMeansGPUOneAPICompute(
    JNIEnv *env, jint rankId, jlong pNumTabData, jlong pNumTabCenters,
    jint cluster_num, jdouble tolerance, jint iteration_num, jint executor_num,
    const ccl::string &ipPort, cl::sycl::queue *gpu_queue, jobject resultObj) {
    std::cout << "oneDAL (native): CPU compute start , rankid %ld " << rankId
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    homogen_table *htable = ((HomogenTablePtr *)pNumTabData)->get();
    homogen_table *centroids = ((HomogenTablePtr *)pNumTabCenters)->get();
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(cluster_num)
                                 .set_max_iteration_count(iteration_num)
                                 .set_accuracy_threshold(tolerance);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        *gpu_queue, executor_num, rankId, ipPort);
    kmeans::train_input local_input{*htable, *centroids};
    auto result_train = preview::train(comm, kmeans_desc, local_input);
    if (isRoot) {
        std::cout << "iteration_num: " << iteration_num << std::endl;
        std::cout << "Iteration count: " << result_train.get_iteration_count()
                  << std::endl;
        std::cout << "Objective function value: "
                  << result_train.get_objective_function_value() << std::endl;
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
#endif

/*
 * Class:     com_intel_oap_mllib_clustering_KMeansDALImpl
 * Method:    cKMeansOneapiComputeWithInitCenters
 * Signature: (JJIDIIILcom/intel/oap/mllib/clustering/KMeansResult;)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_clustering_KMeansDALImpl_cKMeansOneapiComputeWithInitCenters(
    JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabCenters,
    jint cluster_num, jdouble tolerance, jint iteration_num, jint executor_num,
    jint compute_device, jint rankId, jstring ip_port, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels with " << std::endl;
    const char *ipport = env->GetStringUTFChars(ip_port, 0);
    std::string ipPort = std::string(ipport);
    jlong ret = 0L;
    switch (getComputeDevice(compute_device)) {
    case compute_device::host: {
        ret = doKMeansHostOneAPICompute(
            env, rankId, pNumTabData, pNumTabCenters, cluster_num, tolerance,
            iteration_num, executor_num, ipPort, nullptr, resultObj);
    }
#ifdef CPU_GPU_PROFILE
    case compute_device::cpu: {
        cout << "oneDAL (native): use DPCPP CPU kernels" << endl;
        cl::sycl::queue *cpu_queue = getQueue(false);
        ret = doKMeansCPUOneAPICompute(
            env, rankId, pNumTabData, pNumTabCenters, cluster_num, tolerance,
            iteration_num, executor_num, ipPort, cpu_queue, resultObj);
    }
    case compute_device::gpu: {
        cout << "oneDAL (native): use DPCPP GPU kernels" << endl;
        cl::sycl::queue *gpu_queue = getQueue(true);
        ret = doKMeansGPUOneAPICompute(
            env, rankId, pNumTabData, pNumTabCenters, cluster_num, tolerance,
            iteration_num, executor_num, ipPort, gpu_queue, resultObj);
    }
#endif
    }
    env->ReleaseStringUTFChars(ip_port, ipport);
    return ret;
}
