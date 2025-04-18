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
                          "LinearRegression first 20 columns of "
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
                          "RidgeRegression first 20 columns of "
                          "coefficients (w0, w1..wn):",
                          1, 20);
    }
    return resultTable;
}

#ifdef CPU_GPU_PROFILE
static jlong doLROneAPICompute(
    JNIEnv *env, size_t rankId,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jlong pNumTabFeature, jlong featureRows, jlong featureCols,
    jlong pNumTabLabel, jlong labelCols, jboolean jfitIntercept,
    jint executorNum, jobject resultObj) {
    logger::println(logger::INFO,
                    "OneDAL (native): GPU compute start , rankid %d", rankId);
    const bool isRoot = (rankId == ccl_root);
    bool fitIntercept = bool(jfitIntercept);

    homogen_table xtrain = *reinterpret_cast<homogen_table *>(
        createHomogenTableWithArrayPtr(pNumTabFeature, featureRows, featureCols,
                                       comm.get_queue())
            .get());
    homogen_table ytrain = *reinterpret_cast<homogen_table *>(
        createHomogenTableWithArrayPtr(pNumTabLabel, featureRows, labelCols,
                                       comm.get_queue())
            .get());

    linear_regression_gpu::train_input local_input{xtrain, ytrain};
    const auto linear_regression_desc =
        linear_regression_gpu::descriptor<GpuAlgorithmFPType>(fitIntercept);
    comm.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();
    linear_regression_gpu::train_result result_train =
        preview::train(comm, linear_regression_desc, xtrain, ytrain);
    if (isRoot) {
        HomogenTablePtr result_matrix = std::make_shared<homogen_table>(
            result_train.get_model().get_betas());
        auto t2 = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration<float>(t2 - t1).count();
        logger::println(
            logger::INFO,
            "LinearRegression (native): training step took %f secs.", duration);
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
    jint executorCores, jint computeDeviceOrdinal, jintArray gpuIdxArray,
    jobject resultObj) {

    logger::println(logger::INFO,
                    "OneDAL (native): use DPC++ kernels; device %s",
                    ComputeDeviceString[computeDeviceOrdinal].c_str());

    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    bool useGPU = false;
    if (device == ComputeDevice::gpu && regParam == 0) {
        useGPU = true;
    }
    NumericTablePtr resultTable;
    jlong resultptr = 0L;
    if (useGPU) {
#ifdef CPU_GPU_PROFILE
        logger::println(logger::INFO,
                        "OneDAL (native): use GPU kernels with rankid %d",
                        rank);
        auto comm = getDalComm();
        resultptr = doLROneAPICompute(env, rank, comm, feature, featureRows,
                                      featureCols, label, labelCols,
                                      fitIntercept, executorNum, resultObj);
#endif
    } else {
        ccl::communicator &cclComm = getComm();
        size_t rankId = cclComm.rank();

        NumericTablePtr pLabel = *((NumericTablePtr *)label);
        NumericTablePtr pData = *((NumericTablePtr *)feature);

        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(executorCores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        logger::println(logger::INFO,
                        "OneDAL (native): Number of CPU threads used %d",
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
        resultptr = (jlong)coeffvectors;
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
