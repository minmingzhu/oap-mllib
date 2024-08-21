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

#ifdef CPU_GPU_PROFILE
#include "Common.hpp"
#include "oneapi/dal/algo/covariance.hpp"
#endif

#ifndef ONEDAL_DATA_CONVERSION
#define ONEDAL_DATA_CONVERSION
#include "data_management/data_source/csv_feature_manager.h"
#include "data_management/data_source/file_data_source.h"
#undef ONEDAL_DATA_CONVERSION
#endif

#include "OneCCL.h"
#include "com_intel_oap_mllib_stat_CorrelationDALImpl.h"
#include "service.h"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "Logger.h"
//#include <cstdlib> // for getenv

using namespace std;
#ifdef CPU_GPU_PROFILE
namespace covariance_gpu = oneapi::dal::covariance;
std::shared_ptr<file_store> store;
#endif
using namespace daal;
using namespace daal::services;
namespace covariance_cpu = daal::algorithms::covariance;
extern char **environ;

static void doCorrelationDaalCompute(JNIEnv *env, jobject obj, size_t rankId,
                                     ccl::communicator &comm,
                                     const NumericTablePtr &pData,
                                     size_t nBlocks, jobject resultObj) {
    using daal::byte;
    auto t1 = std::chrono::high_resolution_clock::now();

    const bool isRoot = (rankId == ccl_root);

    covariance_cpu::Distributed<step1Local, CpuAlgorithmFPType> localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(covariance_cpu::data, pData);

    /* Compute covariance */
    localAlgorithm.compute();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    logger::println(logger::INFO,
                    "Correleation (native): local step took %d secs",
                    duration / 1000);

    t1 = std::chrono::high_resolution_clock::now();

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
    t2 = std::chrono::high_resolution_clock::now();

    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    logger::println(logger::INFO,
                    "Correleation (native): ccl_allgatherv took %d secs",
                    duration / 1000);
    if (isRoot) {
        auto t1 = std::chrono::high_resolution_clock::now();
        /* Create an algorithm to compute covariance on the master node */
        covariance_cpu::Distributed<step2Master, CpuAlgorithmFPType>
            masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            covariance_cpu::PartialResultPtr dataForStep2FromStep1(
                new covariance_cpu::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm
             */
            masterAlgorithm.input.add(covariance_cpu::partialResults,
                                      dataForStep2FromStep1);
        }

        /* Set the parameter to choose the type of the output matrix */
        masterAlgorithm.parameter.outputMatrixType =
            covariance_cpu::correlationMatrix;

        /* Merge and finalizeCompute covariance decomposition on the master node
         */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        covariance_cpu::ResultPtr result = masterAlgorithm.getResult();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        logger::println(logger::INFO,
                        "Correleation (native): master step took %d secs",
                        duration / 1000);

        /* Print the results */
        printNumericTable(result->get(covariance_cpu::correlation),
                          "Correlation first 20 columns of "
                          "correlation matrix:",
                          1, 20);
        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID correlationNumericTableField =
            env->GetFieldID(clazz, "correlationNumericTable", "J");

        NumericTablePtr *correlation =
            new NumericTablePtr(result->get(covariance_cpu::correlation));

        env->SetLongField(resultObj, correlationNumericTableField,
                          (jlong)correlation);
    }
}

std::vector<std::string> file_path(const std::string& path) {
    std::vector<std::string> result;
    for (auto& file : fs::directory_iterator(path)){
         if(fs::is_empty(file.path())){
             continue;
         }else if(file.path().extension()==".crc" || file.path().extension()==""){
             continue;
         }else{
            result.push_back(file.path());
         }
    }
    return result;
}

#ifdef CPU_GPU_PROFILE
static jlong doCorrelationOneAPICompute(
    JNIEnv *env, jlong pNumTabData, long numRows, long numClos,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj, sycl::queue &queue, std::string breakdown_name) {
    logger::println(logger::INFO, "oneDAL (native): GPU compute start");
    const char* env_var = std::getenv("ZE_AFFINITY_MASK"); // replace "PATH" with the environment variable you want to check
    const char* env_var_1 = std::getenv("ZE_ENABLE_PCI_ID_DEVICE_ORDER"); // replace "PATH" with the environment variable you want to check

    if (env_var) {
        std::cout << "ZE_AFFINITY_MASK: " << env_var << std::endl;
        std::cout << "ZE_ENABLE_PCI_ID_DEVICE_ORDER: " << env_var_1 << std::endl;
    } else {
        std::cout << "Environment variable not found." << std::endl;
    }
//    for (char **env = environ; *env != nullptr; ++env) {
//        std::cout << *env << std::endl;
//    }
    const bool isRoot = (comm.get_rank() == ccl_root);
//    auto t1 = std::chrono::high_resolution_clock::now();
//    auto input_vec = file_path("/home/damon/storage/DataRoot/HiBench_CSV/Correlation/Input/4000000");
//    const auto train_data_file_name = data_path(input_vec[comm.get_rank()]);
//    cout << "rank id = " << comm.get_rank()  << " File name: " << train_data_file_name << endl;
//    const auto htable = read<table>(queue, csv::data_source{ train_data_file_name });
//    comm.barrier();

    float *htableArray = reinterpret_cast<float *>(pNumTabData);
    logger::println(logger::INFO, "numRows was %d", numRows);
    logger::println(logger::INFO, "numClos was %d", numClos);
    auto t1 = std::chrono::high_resolution_clock::now();

//    auto data = sycl::malloc_shared<float>(numRows * numClos, queue);
//    std::cout << "table size : " << numRows * numClos << std::endl;
//    logger::Logger::getInstance(breakdown_name).printLogToFile("rankID was %d, table size %ld.", comm.get_rank(), numRows * numClos );
//    queue.memcpy(data, htableArray, sizeof(float) * numRows * numClos).wait();
    auto data =
        oneapi::dal::array<float>::empty(queue, numRows * numClos, sycl::usm::alloc::device);

    detail::memcpy_host2usm(queue,
                                 data.get_mutable_data(),
                                 htableArray,
                                 sizeof(float) * numRows * numClos);

//    table htable = detail::homogen_table_builder{}.reset(data, numRows, numClos).build();
    homogen_table htable = homogen_table::wrap(data, numRows, numClos);

    freeArrayPtr<float>(htableArray);
//    homogen_table htable{queue, data, numRows, numClos,
//                         detail::make_default_delete<const float>(queue)};
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    logger::println(logger::INFO,
                   "Correlation batch(native): create homogen table took %f secs",
                   duration / 1000);

    logger::Logger::getInstance(breakdown_name).printLogToFile("rankID was %d, create homogen table took %f secs.", comm.get_rank(), duration / 1000 );

    const auto cor_desc =
        covariance_gpu::descriptor<GpuAlgorithmFPType>{}.set_result_options(
            covariance_gpu::result_options::cor_matrix | covariance_gpu::result_options::means);
    comm.barrier();
    t1 = std::chrono::high_resolution_clock::now();
    logger::println(logger::INFO, "Correlation batch(native): compute start");
    const auto result_train = preview::compute(comm, cor_desc, htable);
    t2 = std::chrono::high_resolution_clock::now();
    duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    logger::println(logger::INFO,
                    "Correlation batch(native): computing step took %f secs.",
                    duration / 1000);

    logger::Logger::getInstance(breakdown_name).printLogToFile("rankID was %d, Correlation computing step took %f secs.", comm.get_rank(), duration / 1000 );
    if (isRoot) {
        t2 = std::chrono::high_resolution_clock::now();
        duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                       t2 - t1)
                       .count();
        logger::println(
            logger::INFO,
            "Correlation batch(native): computing step took %f secs.",
            duration / 1000);
        logger::Logger::getInstance(breakdown_name).printLogToFile("rankID was %d, training step took %f secs.", comm.get_rank(), duration / 1000 );
        HomogenTablePtr correlation =
            std::make_shared<homogen_table>(result_train.get_cor_matrix());
        saveHomogenTablePtrToVector(correlation);

        // Return all covariance & mean
//        jclass clazz = env->GetObjectClass(resultObj);
//
//        // Get Field references
//        jfieldID correlationNumericTableField =
//            env->GetFieldID(clazz, "correlationNumericTable", "J");


//
//        env->SetLongField(resultObj, correlationNumericTableField,
//                          (jlong)correlation.get());
        return (jlong)correlation.get();
    } else {
        return (jlong)0;
    }
}
#endif

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_CorrelationDALImpl_cCorrelationTrainDAL(
    JNIEnv *env, jobject obj, jint rank, jlong pNumTabData, jlong numRows, jlong numClos,
    jint executorNum, jint executorCores, jint computeDeviceOrdinal,
    jintArray gpuIdxArray, jstring ip_port, jstring breakdown_name, jstring store_path, jobject resultObj) {
    logger::println(logger::INFO,
                    "oneDAL (native): use DPC++ kernels; device %s",
                    ComputeDeviceString[computeDeviceOrdinal].c_str());
    jlong ret = 0L;
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch (device) {
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        ccl::communicator &cclComm = getComm();
        int rankId = cclComm.rank();
        NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(executorCores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        logger::println(logger::INFO,
                        "oneDAL (native): Number of CPU threads used %d",
                        nThreadsNew);
        doCorrelationDaalCompute(env, obj, rankId, cclComm, pData, executorNum,
                                 resultObj);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case ComputeDevice::gpu: {
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
        store = std::make_shared<file_store>(
                kvs_store_path, rank, std::chrono::seconds(STORE_TIMEOUT_SEC));
        logger::println(logger::INFO, "create_main_kvs");
        if (create_kvs_by_store(store, rank, kvs, c_breakdown_name) != KVS_CREATE_SUCCESS) {
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
//        auto gpu_device = sycl::device(sycl::gpu_selector_v);
//        sycl::queue queue{gpu_device};
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
        ret = doCorrelationOneAPICompute(env, pNumTabData, numRows, numClos, comm,
                                       resultObj, queue, c_breakdown_name);

        env->ReleaseIntArrayElements(gpuIdxArray, gpuIndices, 0);
        env->ReleaseStringUTFChars(breakdown_name, cstr);
        env->ReleaseStringUTFChars(ip_port, str);
        env->ReleaseStringUTFChars(store_path, path);
        break;
    }
#endif
    default: {
        deviceError("Correlation",
                    ComputeDeviceString[computeDeviceOrdinal].c_str());
    }
    }
    return ret;
}
