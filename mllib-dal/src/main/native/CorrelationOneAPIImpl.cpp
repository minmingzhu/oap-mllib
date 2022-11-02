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
#include "GPU.h"
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "OutputHelpers.hpp"
#include "com_intel_oap_mllib_stat_CorrelationDALImpl.h"
#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;

static void doCorrelationOneAPICompute(JNIEnv *env, jint rankId,
                                       jlong pNumTabData, jint executorNum,
                                       const ccl::string &ipPort,
                                       jint computeDeviceOrdinal,
                                       jobject resultObj) {
    std::cout << "oneDAL (native): compute start , rankid = " << rankId
              << "; device = " << ComputeDeviceString[computeDeviceOrdinal]
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);
    std::cout <<"htable :" <<  htable << std::endl;
    std::cout <<"htable get_row_count() :" <<  htable.get_row_count() << std::endl;
    std::cout <<"htable get_column_count():" <<  htable.get_column_count() << std::endl;

    std::cout << "read start" << std::endl;
    auto device1 = sycl::gpu_selector{}.select_device();
    sycl::queue q{ device1 };
//    const auto input = read<table>(q, csv::data_source{ input_file_name });
    const auto cor_desc = covariance::descriptor{}.set_result_options(
        covariance::result_options::cor_matrix |
        covariance::result_options::means);
//    auto queue = getQueue(device);

//    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(queue);
//    auto rank_id = comm.get_rank();
//    auto rank_count = comm.get_rank_count();
//    std::cout <<"rank_id :" <<  comm.get_rank() << std::endl;
//    std::cout <<"rank_count :" <<  comm.get_rank_count() << std::endl;
    std::cout <<"start:" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    const auto result_train = compute(q, cor_desc, htable);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
                std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "Correlation (native) RankId = << " << rankId
                  << "; spend training times : " << duration
                      << " secs" << std::endl;
    if (isRoot) {
        std::cout << "Mean:\n" << result_train.get_means() << std::endl;
        std::cout << "Correlation:\n"
                  << result_train.get_cor_matrix() << std::endl;
        t2 = std::chrono::high_resolution_clock::now();
        duration =
                  std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        std::cout << "Correlation batch(native) spend training times : " << duration
                        << " secs" << std::endl;
        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID correlationNumericTableField =
            env->GetFieldID(clazz, "correlationNumericTable", "J");

        HomogenTablePtr correlation =
            std::make_shared<homogen_table>(result_train.get_cor_matrix());
        saveHomogenTablePtrToVector(correlation);

        env->SetLongField(resultObj, correlationNumericTableField,
                          (jlong)correlation.get());
    }
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_CorrelationDALImpl_cCorrelationTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executorNum,
    jint computeDeviceOrdinal, jint rankId, jstring ipPort, jobject resultObj) {
    std::cout << "oneDAL (native): use DPC++ kernels " << std::endl;
    const char *ipPortPtr = env->GetStringUTFChars(ipPort, 0);
    std::string ipPortStr = std::string(ipPortPtr);
    doCorrelationOneAPICompute(env, rankId, pNumTabData, executorNum, ipPortStr,
                               computeDeviceOrdinal, resultObj);

    env->ReleaseStringUTFChars(ipPort, ipPortPtr);
    return 0;
}
#endif
