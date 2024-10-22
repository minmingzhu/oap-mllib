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
#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <vector>

#ifdef CPU_GPU_PROFILE
#include "Common.hpp"
#include "oneapi/dal/algo/linear_regression.hpp"
#endif

#include "Logger.h"
#include "OneCCL.h"
#include "com_intel_oap_mllib_regression_LinearRegressionDALImpl.h"
#include "service.h"

using namespace std;
#ifdef CPU_GPU_PROFILE
namespace linear_regression_gpu = oneapi::dal::linear_regression;
std::shared_ptr<file_store> store_lr;
#endif
using namespace daal;
using namespace daal::services;
namespace linear_regression_cpu = daal::algorithms::linear_regression;
namespace ridge_regression_cpu = daal::algorithms::ridge_regression;

static NumericTablePtr linear_regression_compute(
    size_t rankId, ccl::communicator &comm, const NumericTablePtr &pData,
    const NumericTablePtr &pLabel, bool fitIntercept, size_t nBlocks) {
    using daal::byte;

    linear_regression_cpu::training::Distributed<step1Local> localAlgorithm;

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(linear_regression_cpu::training::data, pData);
    localAlgorithm.input.set(
        linear_regression_cpu::training::dependentVariables, pLabel);
    localAlgorithm.parameter.interceptFlag = fitIntercept;

    /* Train the multiple linear regression model on local nodes */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    serializedData =
        services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);

    byte *nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);
    std::vector<size_t> aReceiveCount(comm.size(),
                                      perNodeArchLength); // 4 x "14016"

    /* Transfer partial results to step 2 on the root node */
    ccl::gather((int8_t *)nodeResults, perNodeArchLength,
                (int8_t *)(serializedData.get()), perNodeArchLength, comm)
        .wait();

    delete[] nodeResults;

    NumericTablePtr resultTable;
    if (rankId == ccl_root) {
        /* Create an algorithm object to build the final multiple linear
         * regression model on the master node */
        linear_regression_cpu::training::Distributed<step2Master>
            masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            linear_regression_cpu::training::PartialResultPtr
                dataForStep2FromStep1 =
                    linear_regression_cpu::training::PartialResultPtr(
                        new linear_regression_cpu::training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local multiple linear regression model as input for the
             * master-node algorithm */
            masterAlgorithm.input.add(
                linear_regression_cpu::training::partialModels,
                dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the multiple linear regression model on the
         * master node */
        masterAlgorithm.parameter.interceptFlag = fitIntercept;
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        linear_regression_cpu::training::ResultPtr trainingResult =
            masterAlgorithm.getResult();
        resultTable =
            trainingResult->get(linear_regression_cpu::training::model)
                ->getBeta();

        printNumericTable(resultTable,
                          "Linear Regression first 20 columns of "
                          "coefficients (w0, w1..wn):",
                          1, 20);
    }
    return resultTable;
}

static NumericTablePtr
ridge_regression_compute(size_t rankId, ccl::communicator &comm,
                         const NumericTablePtr &pData,
                         const NumericTablePtr &pLabel, bool fitIntercept,
                         double regParam, size_t nBlocks) {

    using daal::byte;

    NumericTablePtr ridgeParams(new HomogenNumericTable<double>(
        1, 1, NumericTable::doAllocate, regParam));
    ridge_regression_cpu::training::Distributed<step1Local> localAlgorithm;
    localAlgorithm.parameter.ridgeParameters = ridgeParams;
    localAlgorithm.parameter.interceptFlag = fitIntercept;

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(ridge_regression_cpu::training::data, pData);
    localAlgorithm.input.set(ridge_regression_cpu::training::dependentVariables,
                             pLabel);

    /* Train the multiple ridge regression model on local nodes */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    serializedData =
        services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);

    byte *nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);
    std::vector<size_t> aReceiveCount(comm.size(),
                                      perNodeArchLength); // 4 x "14016"

    /* Transfer partial results to step 2 on the root node */
    ccl::gather((int8_t *)nodeResults, perNodeArchLength,
                (int8_t *)(serializedData.get()), perNodeArchLength, comm)
        .wait();

    delete[] nodeResults;

    NumericTablePtr resultTable;
    if (rankId == ccl_root) {
        /* Create an algorithm object to build the final multiple ridge
         * regression model on the master node */
        ridge_regression_cpu::training::Distributed<step2Master>
            masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            ridge_regression_cpu::training::PartialResultPtr
                dataForStep2FromStep1 =
                    ridge_regression_cpu::training::PartialResultPtr(
                        new ridge_regression_cpu::training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local multiple ridge regression model as input for the
             * master-node algorithm */
            masterAlgorithm.input.add(
                ridge_regression_cpu::training::partialModels,
                dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the multiple ridge regression model on the
         * master node */
        masterAlgorithm.parameter.interceptFlag = fitIntercept;
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        ridge_regression_cpu::training::ResultPtr trainingResult =
            masterAlgorithm.getResult();
        resultTable = trainingResult->get(ridge_regression_cpu::training::model)
                          ->getBeta();

        printNumericTable(resultTable,
                          "Ridge Regression first 20 columns of "
                          "coefficients (w0, w1..wn):",
                          1, 20);
    }
    return resultTable;
}

#ifdef CPU_GPU_PROFILE
static jlong doLROneAPICompute(JNIEnv *env, size_t rankId,
                               preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
                               sycl::queue &queue,
                               jlong pNumTabFeature, jlong featureRows,
                               jlong featureCols, jlong pNumTabLabel,
                               jlong labelCols, jboolean jfitIntercept,
                               jint executorNum, jobject resultObj,
                               std::string  breakdown_name) {
    logger::println(logger::INFO,
                    "oneDAL (native): GPU compute start , rankid %d", rankId);
    const bool isRoot = (rankId == ccl_root);
    bool fitIntercept = bool(jfitIntercept);

    float *htableFeatureArray = reinterpret_cast<float *>(pNumTabFeature);
    float *htableLabelArray = reinterpret_cast<float *>(pNumTabLabel);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto featureData =
        sycl::malloc_shared<float>(featureRows * featureCols, queue);
    queue
        .memcpy(featureData, htableFeatureArray,
                sizeof(float) * featureRows * featureCols)
        .wait();
    homogen_table xtrain{queue, featureData, featureRows, featureCols,
                         detail::make_default_delete<const float>(queue)};
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    logger::println(logger::INFO,
                   "LinerRegression(native): create feature homogen table took %f secs",
                   duration / 1000);
    logger::Logger::getInstance(breakdown_name).printLogToFile("rankID was %d, create homogen table took %f secs.", comm.get_rank(), duration / 1000 );

    t1 = std::chrono::high_resolution_clock::now();
    auto labelData =
        sycl::malloc_shared<float>(featureRows * labelCols, queue);
    queue
        .memcpy(labelData, htableLabelArray,
                sizeof(float) * featureRows * labelCols)
        .wait();
    homogen_table ytrain{queue, labelData, featureRows, labelCols,
                         detail::make_default_delete<const float>(queue)};
    t2 = std::chrono::high_resolution_clock::now();
    duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    logger::println(logger::INFO,
                   "LinerRegression(native): create label homogen table took %f secs",
                   duration / 1000);
    linear_regression_gpu::train_input local_input{xtrain, ytrain};
    const auto linear_regression_desc =
        linear_regression_gpu::descriptor<GpuAlgorithmFPType>(fitIntercept);
    comm.barrier();
    t1 = std::chrono::high_resolution_clock::now();

    linear_regression_gpu::train_result result_train =
        preview::train(comm, linear_regression_desc, xtrain, ytrain);
//    t2 = std::chrono::high_resolution_clock::now();
//    duration =
//        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
//                                                                     t1)
//            .count();
//    logger::Logger::getInstance(breakdown_name).printLogToFile("rankID was %d, LinerRegression training step took %f secs.", comm.get_rank(), duration / 1000 );
    if (isRoot) {
        HomogenTablePtr result_matrix = std::make_shared<homogen_table>(
            result_train.get_model().get_betas());
        t2 = std::chrono::high_resolution_clock::now();
        duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
                                                                         t1)
                .count();
        logger::println(logger::INFO,
                       "LinerRegression(native): training step took %f secs",
                       duration / 1000);
        logger::Logger::getInstance(breakdown_name).printLogToFile("rankID was %d, training step took %f secs.", comm.get_rank(), duration / 1000 );

        saveHomogenTablePtrToVector(result_matrix);
        return (jlong)result_matrix.get();
    } else {
        return (jlong)0;
    }
}
#endif

/*
 * Class:     com_intel_oap_mllib_regression_LinearRegressionDALImpl
 * Method:    cLinearRegressionTrainDAL
 * Signature: (JJZDDIII[ILcom/intel/oap/mllib/regression/LiRResult;)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_regression_LinearRegressionDALImpl_cLinearRegressionTrainDAL(
    JNIEnv *env, jobject obj, jint rank, jlong feature, jlong featureRows,
    jlong featureCols, jlong label, jlong labelCols, jboolean fitIntercept,
    jdouble regParam, jdouble elasticNetParam, jint executorNum,
    jint executorCores, jint computeDeviceOrdinal, jintArray gpuIdxArray, jstring ip_port, jstring breakdown_name,
    jstring store_path, jobject resultObj) {

    logger::println(logger::INFO,
                    "oneDAL (native): use DPC++ kernels; device %s",
                    ComputeDeviceString[computeDeviceOrdinal].c_str());



    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    bool useGPU = false;
    if (device == ComputeDevice::gpu && regParam == 0) {
        useGPU = true;
    }
    jlong resultptr = 0L;
    if (useGPU) {
#ifdef CPU_GPU_PROFILE
        int nGpu = env->GetArrayLength(gpuIdxArray);
        logger::println(
            logger::INFO,
            "oneDAL (native): use GPU kernels with %d GPU(s) rankid %d", nGpu,
            rank);

        jint *gpuIndices = env->GetIntArrayElements(gpuIdxArray, 0);
        auto gpus = get_gpus();
        const char* cstr = env->GetStringUTFChars(breakdown_name, nullptr);
        std::string c_breakdown_name(cstr);
        const char *str = env->GetStringUTFChars(ip_port, nullptr);
        ccl::string ccl_ip_port(str);
        const char* path = env->GetStringUTFChars(store_path, 0);
        std::string kvs_store_path(path);
        ccl::shared_ptr_class<ccl::kvs> kvs;
        auto t1 = std::chrono::high_resolution_clock::now();

        ccl::init();

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        logger::println(logger::INFO, "OneCCL singleton init took %f secs",
                        duration / 1000);
        logger::Logger::getInstance(c_breakdown_name).printLogToFile("rankID was %d, OneCCL singleton init took %f secs.", rank, duration / 1000 );


        t1 = std::chrono::high_resolution_clock::now();
        store_lr = std::make_shared<file_store>(
                kvs_store_path, rank, std::chrono::seconds(STORE_TIMEOUT_SEC));
        logger::println(logger::INFO, "create_main_kvs");
        if (create_kvs_by_store(store_lr, rank, kvs, c_breakdown_name) != KVS_CREATE_SUCCESS) {
            logger::println(logger::INFO, "can not create kvs by store");
            return -1;
        }
//        auto kvs_attr = ccl::create_kvs_attr();
//
//        kvs_attr.set<ccl::kvs_attr_id::ip_port>(ccl_ip_port);
//
//        ccl::shared_ptr_class<ccl::kvs> kvs = ccl::create_main_kvs(kvs_attr);

        t2 = std::chrono::high_resolution_clock::now();
        duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        logger::println(logger::INFO, "OneCCL (native): create kvs took %f secs",
                        duration / 1000);
        logger::Logger::getInstance(c_breakdown_name).printLogToFile("rankID was %d, OneCCL create communicator took %f secs.", rank, duration / 1000 );
        sycl::queue queue{gpus[0]};
        t1 = std::chrono::high_resolution_clock::now();
        auto comm =
            preview::spmd::make_communicator<preview::spmd::backend::ccl>(
                queue, executorNum, rank, kvs);
        t2 = std::chrono::high_resolution_clock::now();
        duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        logger::Logger::getInstance(c_breakdown_name).printLogToFile("rankID was %d, create communicator took %f secs.", rank, duration / 1000 );
        resultptr = doLROneAPICompute(
            env, rank, comm, queue, feature, featureRows, featureCols,
            label, labelCols, fitIntercept, executorNum, resultObj, c_breakdown_name);
        env->ReleaseIntArrayElements(gpuIdxArray, gpuIndices, 0);
        env->ReleaseStringUTFChars(breakdown_name, cstr);
        env->ReleaseStringUTFChars(ip_port, str);
        env->ReleaseStringUTFChars(store_path, path);

#endif
    } else {
        ccl::communicator &cclComm = getComm();
        size_t rankId = cclComm.rank();
        NumericTablePtr resultTable;
        NumericTablePtr pLabel = *((NumericTablePtr *)label);
        NumericTablePtr pData = *((NumericTablePtr *)feature);

        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(executorCores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        logger::println(logger::INFO,
                        "oneDAL (native): Number of CPU threads used %d",
                        nThreadsNew);
        if (regParam == 0) {
            resultTable = linear_regression_compute(
                rankId, cclComm, pData, pLabel, fitIntercept, executorNum);
        } else {
            resultTable =
                ridge_regression_compute(rankId, cclComm, pData, pLabel,
                                         fitIntercept, regParam, executorNum);
        }

        NumericTablePtr *coeffvectors = new NumericTablePtr(resultTable);
        if (rankId == ccl_root) {
            // Get the class of the result object
            jclass clazz = env->GetObjectClass(resultObj);
            // Get Field references
            jfieldID coeffNumericTableField =
                env->GetFieldID(clazz, "coeffNumericTable", "J");

            env->SetLongField(resultObj, coeffNumericTableField, resultptr);

            // intercept is already in first column of coeffvectors
            resultptr = (jlong)coeffvectors;
        }
    }
    return resultptr;
}
