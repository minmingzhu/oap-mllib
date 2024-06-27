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

#pragma once
#include <iostream>
#include <mutex>
#include <vector>
#include <oneapi/ccl.hpp>

#include "Logger.h"
static std::vector<ccl::communicator> g_ccl_comms;
static std::vector<ccl::shared_ptr_class<ccl::kvs>> g_ccl_kvs;
class CCLInitSingleton {
public:
    static CCLInitSingleton& get(int size, int rank, ccl::string ccl_ip_port, ccl::string name) {
        static std::once_flag flag;
        static CCLInitSingleton instance;
        std::call_once(flag, [size, rank, ccl_ip_port, name] {
            instance = CCLInitSingleton(size, rank, ccl_ip_port, name);
        });
        return instance;
    }
    ccl::communicator& getComm() {
        return g_ccl_comms[0];
    }

    ccl::shared_ptr_class<ccl::kvs>& getKvs() {
        return g_ccl_kvs[0];
    }
private:
    CCLInitSingleton() {
    }

    CCLInitSingleton(int size, int rank, ccl::string ccl_ip_port, ccl::string name) {
        auto t1 = std::chrono::high_resolution_clock::now();

        ccl::init();

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        logger::println(logger::INFO, "OneCCL singleton init took %f secs",
                        duration / 1000);
        logger::Logger::getInstance(name).printLogToFile("rankID was %d, OneCCL singleton init took %f secs.", rank, duration / 1000 );


        t1 = std::chrono::high_resolution_clock::now();
        auto kvs_attr = ccl::create_kvs_attr();
        kvs_attr.set<ccl::kvs_attr_id::ip_port>(ccl_ip_port);

        auto kvs = ccl::create_main_kvs(kvs_attr);
        g_ccl_kvs.push_back(kvs);
        g_ccl_comms.push_back(std::move(ccl::create_communicator(size, rank, kvs)));
        t2 = std::chrono::high_resolution_clock::now();
        duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        logger::println(logger::INFO, "OneCCL (native): init took %f secs",
                        duration / 1000);
        logger::Logger::getInstance(name).printLogToFile("rankID was %d, OneCCL create communicator took %f secs.", rank, duration / 1000 );

}
};
