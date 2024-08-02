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

#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "Communicator.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/common.hpp"
#include "com_intel_oap_mllib_stat_CorrelationDALImpl.h"


namespace dal = oneapi::dal;
using namespace std;
namespace fs = std::filesystem;

std::ostream &operator<<(std::ostream &stream, const oneapi::dal::table &table) {
    std::cout << "output : " << std::endl;
    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    if (table.get_row_count() <= 10) {
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            if(table.get_column_count() <= 20) {
                for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                    std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                              << std::setprecision(6) << x[i * table.get_column_count() + j];
                }
                std::cout << std::endl;
            } else {
                for (std::int64_t j = 0; j < 20; j++) {
                    std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                              << std::setprecision(6) << x[i * table.get_column_count() + j];
                }
                std::cout << std::endl;
            }
        }
    }
    else {
        for (std::int64_t i = 0; i < 5; i++) {
            if(table.get_column_count() <= 20) {
                for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                    std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                              << std::setprecision(6) << x[i * table.get_column_count() + j];
                }
                std::cout << std::endl;
            } else {
                for (std::int64_t j = 0; j < 20; j++) {
                    std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                              << std::setprecision(6) << x[i * table.get_column_count() + j];
                }
                std::cout << std::endl;
            }
        }
        std::cout << "..." << (table.get_row_count() - 10) << " lines skipped..." << std::endl;
        for (std::int64_t i = table.get_row_count() - 5; i < table.get_row_count(); i++) {
            if(table.get_column_count() <= 20) {
                for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                    std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                              << std::setprecision(6) << x[i * table.get_column_count() + j];
                }
                std::cout << std::endl;
            } else {
                for (std::int64_t j = 0; j < 20; j++) {
                    std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                              << std::setprecision(6) << x[i * table.get_column_count() + j];
                }
                std::cout << std::endl;
            }
        }
    }
    return stream;
}

inline bool check_file(const std::string& name) {
    return std::ifstream{ name }.good();
}

ccl::shared_ptr_class<ccl::kvs> getCclPortKvs(ccl::string ccl_ip_port){
    std::cout << "ccl_ip_port = " << ccl_ip_port << std::endl;
    auto kvs_attr = ccl::create_kvs_attr();
    std::cout << "ccl_ip_port 1 "<< std::endl;
    kvs_attr.set<ccl::kvs_attr_id::ip_port>(ccl_ip_port);
    std::cout << "ccl_ip_port 2 "<< std::endl;

    ccl::shared_ptr_class<ccl::kvs>  kvs = ccl::create_main_kvs(kvs_attr);
    std::cout << "ccl_ip_port 3 "<< std::endl;

    return kvs;
}

std::vector<sycl::device> get_gpus()
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

inline std::string get_data_path(const std::string& name) {
    const std::vector<std::string> paths = { "./data", "samples/oneapi/dpc/mpi/data" };

    for (const auto& path : paths) {
        const std::string try_path = path + "/" + name;
        if (check_file(try_path)) {
            return try_path;
        }
    }

    return name;
}

std::vector<std::string> get_file_path(const std::string& path) {
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

int getLocalRank(int size, int rank, ccl::communicator& comm)
{
    /* Obtain local rank among nodes sharing the same host name */
    char zero = static_cast<char>(0);
    std::vector<char> name(MPI_MAX_PROCESSOR_NAME + 1, zero);
    int resultlen = 0;
    std::string str(name.begin(), name.end());
    std::vector<char> allNames((MPI_MAX_PROCESSOR_NAME + 1) * size, zero);
    std::vector<size_t> aReceiveCount(size, MPI_MAX_PROCESSOR_NAME + 1);
    ccl::allgatherv((int8_t *)name.data(), name.size(), (int8_t *)allNames.data(), aReceiveCount, comm).wait();
    int localRank = 0;
    for (int i = 0; i < rank; i++)
    {
        auto nameBegin = allNames.begin() + i * (MPI_MAX_PROCESSOR_NAME + 1);
        std::string nbrName(nameBegin, nameBegin + (MPI_MAX_PROCESSOR_NAME + 1));
        if (nbrName == str) localRank++;
    }
    return localRank;
}

void weak(sycl::queue& queue, const string& path, dal::preview::spmd::communicator<dal::preview::spmd::device_memory_access::usm>& comm) {

    const auto cov_desc = dal::covariance::descriptor{}.set_result_options(
        dal::covariance::result_options::cor_matrix | dal::covariance::result_options::means);

    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto input_vec = get_file_path(path);
    const auto train_data_file_name = get_data_path(input_vec[rank_id]);
    cout <<"RankID = " << rank_id  << " File name: " << train_data_file_name << endl;
    auto t1 = chrono::high_resolution_clock::now();
    const auto x_train = dal::read<dal::table>(queue, dal::csv::data_source{ train_data_file_name });

    auto rows = x_train.get_row_count();
    auto cols = x_train.get_column_count();
    auto size = rows * cols;
    cout <<"RankID = " << rank_id  << ", table size " << size << endl;
    comm.barrier();
    // MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = chrono::high_resolution_clock::now();

    cout <<"RankID = " << rank_id  << ", loading CSV took "
         << (float)chrono::duration_cast<chrono::milliseconds>(
                t2 - t1)
                    .count() /
                1000
         << " secs" << endl;
    t1 = chrono::high_resolution_clock::now();
    const auto result = dal::preview::compute(comm, cov_desc, x_train);
    t2 = chrono::high_resolution_clock::now();
    cout <<"RankID = " << rank_id  << ", cov training step took "
        << (float)chrono::duration_cast<chrono::milliseconds>(
                t2 - t1)
                    .count() /
                1000
        << " secs" << endl;
    if(comm.get_rank() == 0) {
        cout << "Mean:\n" << result.get_means() << endl;
        cout << "Correlation:\n" << result.get_cor_matrix() << endl;
        t2 = chrono::high_resolution_clock::now();
        cout <<"RankID = " << rank_id  << ", training step took "
            << (float)chrono::duration_cast<chrono::milliseconds>(
                    t2 - t1)
                        .count() /
                    1000
            << " secs" << endl;
    }
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_CorrelationDALImpl_cCorrelationSampleTrainDAL(
    JNIEnv *env, jobject obj, jint rank, jint rank_count, jstring ip_port){
    cout << "sample`" << endl;
    auto gpus = get_gpus();
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
//    auto ccl_comm = ccl::create_communicator(rank_count, rank, kvs);
//    auto local_rank = getLocalRank(rank_count, rank, ccl_comm);
//    auto rank_id = local_rank % gpus.size();
    t2 = chrono::high_resolution_clock::now();
    cout << "RankID = " << rank
         << ", OneCCL create kvs took "
         << (float)chrono::duration_cast<chrono::milliseconds>(
                t2 - t1)
                    .count() /
                1000
         << " secs" << endl;
    auto device   = gpus[0];
    sycl::queue q{ device };
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
