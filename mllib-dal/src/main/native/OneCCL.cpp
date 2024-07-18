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
#include <iostream>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <ifaddrs.h>
#include <list>
#include <netdb.h>

#include <oneapi/ccl.hpp>

#include "CCLInitSingleton.hpp"
#include "Logger.h"
#include "OneCCL.h"
#include "com_intel_oap_mllib_OneCCL__.h"
#include "store.hpp"

#define STORE_TIMEOUT_SEC 120
#define KVS_CREATE_SUCCESS 0
#define KVS_CREATE_FAILURE -1

extern const size_t ccl_root = 0;

static const int CCL_IP_LEN = 128;
static std::list<std::string> local_host_ips;
static size_t comm_size = 0;
static size_t rank_id = 0;
std::vector<ccl::communicator> g_comms;
std::vector<ccl::shared_ptr_class<ccl::kvs>> g_kvs;

ccl::communicator &getComm() { return g_comms[0]; }
ccl::shared_ptr_class<ccl::kvs> &getKvs() { return g_kvs[0]; }
std::shared_ptr<file_store> store;

static int create_kvs_by_store(std::shared_ptr<file_store> store,
                        int rank,
                        ccl::shared_ptr_class<ccl::kvs>& kvs,
                        ccl::string name) {
    logger::println(logger::INFO, "OneCCL (native): create_kvs_by_store ");
    auto t1 = std::chrono::high_resolution_clock::now();
    ccl::kvs::address_type main_addr;
    auto start = std::chrono::system_clock::now();
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        if (store->write((void*)main_addr.data(), main_addr.size()) < 0) {
            logger::println(logger::INFO, "OneCCL (native): error occurred during write attempt");
            kvs.reset();
            return KVS_CREATE_FAILURE;
        }
        auto end = std::chrono::system_clock::now();
        auto exec_time =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(end -start)
                .count();
        logger::println(logger::INFO, "OneCCL (native): write to store time %f secs",
                exec_time / 1000);
    }
    else {
        if (store->read((void*)main_addr.data(), main_addr.size()) < 0) {
            logger::println(logger::INFO, "OneCCL (native): error occurred during read attempt");
            kvs.reset();
            return KVS_CREATE_FAILURE;
        }
        auto end = std::chrono::system_clock::now();
        auto exec_time =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(end -start)
                .count();
        logger::println(logger::INFO, "OneCCL (native): read from store time %f secs",
                exec_time / 1000);
        kvs = ccl::create_kvs(main_addr);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
    logger::Logger::getInstance(name).printLogToFile("rankID was %d, OneCCL create communicator took %f secs.", rank, duration / 1000 );
    return KVS_CREATE_SUCCESS;
}

JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_c_1init(
    JNIEnv *env, jobject obj, jint size, jint rank, jstring ip_port, jstring name, jstring store_path,
    jobject param) {

    logger::println(logger::INFO, "OneCCL (native): init; Rank id %d", rank);

    ccl::shared_ptr_class<ccl::kvs> kvs;

    const char *str_name = env->GetStringUTFChars(name, 0);
    ccl::string ccl_name(str_name);
    const char* path = env->GetStringUTFChars(store_path, 0);
    std::string kvs_store_path(path);
    store = std::make_shared<file_store>(
                kvs_store_path, rank, std::chrono::seconds(STORE_TIMEOUT_SEC));

    auto t1 = std::chrono::high_resolution_clock::now();
    ccl::init();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    logger::println(logger::INFO, "OneCCL singleton init took %f secs",
                    duration / 1000);
    logger::Logger::getInstance(ccl_name).printLogToFile("rankID was %d, OneCCL singleton init took %f secs.", rank, duration / 1000 );
    logger::println(logger::INFO, "OneCCL (native): create_main_kvs");
    if (create_kvs_by_store(store, rank, kvs, ccl_name) != KVS_CREATE_SUCCESS) {
        logger::println(logger::INFO, "OneCCL (native): can not create kvs by store");
        return -1;
    }

    t1 = std::chrono::high_resolution_clock::now();
    logger::println(logger::INFO, "OneCCL (native): create_kvs_attr");
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_kvs.push_back(kvs);
    }
//    logger::println(logger::INFO, "OneCCL (native): ccl::create_communicator(size, rank, kvs)");
//    logger::println(logger::INFO, "ccl::create_communicator %d ,%d", size, rank);
//    {
//        std::lock_guard<std::mutex> lock(g_mtx);
//        g_comms.push_back(ccl::create_communicator(size, rank, kvs));
//    }
//    logger::println(logger::INFO, "OneCCL (native): ccl::create_communicator finished");

    t2 = std::chrono::high_resolution_clock::now();
    duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    logger::println(logger::INFO, "OneCCL (native): init took %f secs",
                    duration / 1000);
    logger::Logger::getInstance(ccl_name).printLogToFile("rankID was %d, OneCCL create communicator took %f secs.", rank, duration / 1000 );
//
//
//    rank_id = getComm().rank();
//    comm_size = getComm().size();
//
//    jclass cls = env->GetObjectClass(param);
//    jfieldID fid_comm_size = env->GetFieldID(cls, "commSize", "J");
//    jfieldID fid_rank_id = env->GetFieldID(cls, "rankId", "J");

    env->SetLongField(param, fid_comm_size, size);
    env->SetLongField(param, fid_rank_id, rank);
    env->ReleaseStringUTFChars(name, str_name);
    env->ReleaseStringUTFChars(store_path, path);
    logger::println(logger::INFO, "OneCCL (native): init finished");

    return 1;
}

/*
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    c_init
 * Signature: ()I
 */
JNIEXPORT jint JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_c_1initDpcpp(JNIEnv *env, jobject) {
    logger::printerrln(logger::INFO, "OneCCL (native): init dpcpp");

    ccl::init();

    return 1;
}

JNIEXPORT void JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_c_1cleanup(JNIEnv *env, jobject obj) {
    logger::printerrln(logger::INFO, "OneCCL (native): cleanup");
    std::cout << "Size after clear: " << g_kvs.size() << ", Capacity: " << g_kvs.capacity() << std::endl;
    g_kvs.clear();
    std::cout << "Size after clear: " << g_kvs.size() << ", Capacity: " << g_kvs.capacity() << std::endl;
    std::cout << "Size after clear: " << g_comms.size() << ", Capacity: " << g_comms.capacity() << std::endl;
    g_comms.clear();
    std::cout << "Size after clear: " << g_comms.size() << ", Capacity: " << g_comms.capacity() << std::endl;
}

JNIEXPORT jboolean JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_isRoot(JNIEnv *env, jobject obj) {

    return getComm().rank() == 0;
}

JNIEXPORT jint JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_rankID(JNIEnv *env, jobject obj) {
    return getComm().rank();
}

JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_setEnv(
    JNIEnv *env, jobject obj, jstring key, jstring value, jboolean overwrite) {

    char *k = (char *)env->GetStringUTFChars(key, NULL);
    char *v = (char *)env->GetStringUTFChars(value, NULL);

    int err = setenv(k, v, overwrite);

    env->ReleaseStringUTFChars(key, k);
    env->ReleaseStringUTFChars(value, v);

    return err;
}

static int fill_local_host_ip() {
    struct ifaddrs *ifaddr, *ifa;
    int family = AF_UNSPEC;
    char local_ip[CCL_IP_LEN];
    if (getifaddrs(&ifaddr) < 0) {
        logger::printerrln(logger::ERROR,
                           "OneCCL (native): can not get host IP");
        return -1;
    }

    const char iface_name[] = "lo";
    local_host_ips.clear();

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;
        if (strstr(ifa->ifa_name, iface_name) == NULL) {
            family = ifa->ifa_addr->sa_family;
            if (family == AF_INET) {
                memset(local_ip, 0, CCL_IP_LEN);
                int res = getnameinfo(
                    ifa->ifa_addr,
                    (family == AF_INET) ? sizeof(struct sockaddr_in)
                                        : sizeof(struct sockaddr_in6),
                    local_ip, CCL_IP_LEN, NULL, 0, NI_NUMERICHOST);
                if (res != 0) {
                    std::string s("OneCCL (native): getnameinfo error > ");
                    s.append(gai_strerror(res));
                    logger::printerrln(logger::ERROR, s);
                    return -1;
                }
                local_host_ips.push_back(local_ip);
            }
        }
    }
    if (local_host_ips.empty()) {
        logger::printerrln(
            logger::ERROR,
            "OneCCL (native): can't find interface to get host IP");
        return -1;
    }

    freeifaddrs(ifaddr);

    return 0;
}

static bool is_valid_ip(char ip[]) {
    if (fill_local_host_ip() == -1) {
        logger::printerrln(logger::ERROR,
                           "OneCCL (native): get local host ip error");
        return false;
    };

    for (std::list<std::string>::iterator it = local_host_ips.begin();
         it != local_host_ips.end(); ++it) {
        if (*it == ip) {
            return true;
        }
    }

    return false;
}

JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_c_1getAvailPort(
    JNIEnv *env, jobject obj, jstring localIP) {

    // start from beginning of dynamic port
    const int port_start_base = 3000;

    char *local_host_ip = (char *)env->GetStringUTFChars(localIP, NULL);

    // check if the input ip is one of host's ips
    if (!is_valid_ip(local_host_ip))
        return -1;

    struct sockaddr_in main_server_address;
    int server_listen_sock;
    in_port_t port = port_start_base;

    if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("OneCCL (native) getAvailPort error!");
        return -1;
    }

    main_server_address.sin_family = AF_INET;
    main_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    main_server_address.sin_port = htons(port);

    // search for available port
    while (bind(server_listen_sock,
                (const struct sockaddr *)&main_server_address,
                sizeof(main_server_address)) < 0) {
        port++;
        main_server_address.sin_port = htons(port);
    }

    close(server_listen_sock);

    env->ReleaseStringUTFChars(localIP, local_host_ip);

    return port;
}
