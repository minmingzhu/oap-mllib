/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <string>
#include <thread>
#include <filesystem>
#include <jni.h>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "com_intel_oap_mllib_stat_CorrelationDALImpl.h"


namespace dal = oneapi::dal;
using namespace std;
namespace fs = std::filesystem;

inline ccl::shared_ptr_class<ccl::kvs> getCclPortKvs(ccl::string ccl_ip_port){
    std::cout << "ccl_ip_port = " << ccl_ip_port << std::endl;
    auto kvs_attr = ccl::create_kvs_attr();
    std::cout << "ccl_ip_port 1 "<< std::endl;
    kvs_attr.set<ccl::kvs_attr_id::ip_port>(ccl_ip_port);
    std::cout << "ccl_ip_port 2 "<< std::endl;

    ccl::shared_ptr_class<ccl::kvs>  kvs = ccl::create_main_kvs(kvs_attr);
    std::cout << "ccl_ip_port 3 "<< std::endl;

    return kvs;
}

inline std::vector<sycl::device> get_gpus()
{

    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            return devices;
        }
    }
    std::cout << "No GPUs!" << std::endl;
    exit(-3);
    return {};
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_CorrelationDALImpl_cCorrelationSampleTrainDAL(
    JNIEnv *env, jobject obj, jint rank, jint rank_count, jstring ip_port){
    cout << "sample`" << endl;
    const char* env_var = std::getenv("ZE_AFFINITY_MASK"); // replace "PATH" with the environment variable you want to check
    const char* env_var_1 = std::getenv("ZE_ENABLE_PCI_ID_DEVICE_ORDER"); // replace "PATH" with the environment variable you want to check

    if (env_var) {
        std::cout << "ZE_AFFINITY_MASK: " << env_var << std::endl;
        std::cout << "ZE_ENABLE_PCI_ID_DEVICE_ORDER: " << env_var_1 << std::endl;
    } else {
        std::cout << "Environment variable not found." << std::endl;
    }

//    auto gpus = get_gpus();
    const char *str = env->GetStringUTFChars(ip_port, nullptr);
    ccl::string ccl_ip_port(str);

    auto t1 = chrono::high_resolution_clock::now();
    ccl::init();
    auto t2 = chrono::high_resolution_clock::now();
    cout << "OneCCL singleton init took "
        << (float)chrono::duration_cast<chrono::milliseconds>(
                t2 - t1)
                    .count() /
                1000
        << " secs" << endl;

    t1 = chrono::high_resolution_clock::now();

    auto kvs = getCclPortKvs(ccl_ip_port);
    t2 = chrono::high_resolution_clock::now();
    cout << "RankID = " << rank
         << ", OneCCL create kvs took "
         << (float)chrono::duration_cast<chrono::milliseconds>(
                t2 - t1)
                    .count() /
                1000
         << " secs" << endl;
//    auto device   = gpus[0];
//    sycl::queue q{ device };
    auto device = sycl::device(sycl::gpu_selector_v);
    sycl::queue q{device};
    t1 = chrono::high_resolution_clock::now();
    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(q, rank_count, rank, kvs);
    t2 = chrono::high_resolution_clock::now();
            cout << "RankID = " << rank
                << ", create communicator took "
                << (float)chrono::duration_cast<chrono::milliseconds>(
                        t2 - t1)
                            .count() /
                        1000
                << " secs" << endl;
    env->ReleaseStringUTFChars(ip_port, str);
    return 0;
}
