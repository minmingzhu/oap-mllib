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
#include <cstring>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <string>
#include <typeinfo>
#include <vector>
#include <mutex>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#endif
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "com_intel_oneapi_dal_table_HomogenTableImpl.h"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

typedef std::shared_ptr<homogen_table> homogenPtr;
typedef std::shared_ptr<table_metadata> metadataPtr;


std::mutex mtx;
std::vector<homogenPtr> cHomogenVector;
std::vector<metadataPtr> cMetaVector;

static void saveShareHomogenPtrVector(const homogenPtr &ptr) {
       mtx.lock();
       cHomogenVector.push_back(ptr);
       mtx.unlock();
}

static void saveShareMetaPtrVector(const metadataPtr &ptr) {
       mtx.lock();
       cMetaVector.push_back(ptr);
       mtx.unlock();
}

static data_layout getDataLayout(jint cLayout) {
    data_layout layout;
    switch (cLayout) {
    case 0:
        layout = data_layout::unknown;
        break;
    case 1:
        layout = data_layout::row_major;
        break;
    case 2:
        layout = data_layout::column_major;
        break;
    }
    return layout;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    iInit
 * Signature: (JJLjava/nio/ByteBuffer;Ljava/lang/Class;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_iInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jintArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable int init \n");
    jint *fData = env->GetIntArrayElements(cData, NULL);
    homogen_table *h_table;
    homogenPtr tablePtr;
    switch(getComputeDevice(cComputeDevice)) {
       case compute_device::host:{
             h_table = new homogen_table(
                                     fData, cRowCount, cColCount, detail::empty_delete<const int>(),
                                     getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:{
             sycl::queue cpu_queue = getQueue(compute_device::cpu);
             auto cpu_data = malloc_shared<int>(cRowCount * cColCount, cpu_queue);
             cpu_queue.memcpy(cpu_data, fData, sizeof(int) * cRowCount * cColCount).wait();
             h_table = new homogen_table(cpu_queue,
                 cpu_data, cRowCount, cColCount, detail::make_default_delete<const int>(cpu_queue),
                 {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
       }
       case compute_device::gpu:{
             sycl::queue gpu_queue = getQueue(compute_device::gpu);
             auto gpu_data = malloc_shared<int>(cRowCount * cColCount, gpu_queue);
             gpu_queue.memcpy(gpu_data, fData, sizeof(int) * cRowCount * cColCount).wait();
             h_table = new homogen_table(gpu_queue,
                gpu_data, cRowCount, cColCount, detail::make_default_delete<const int>(gpu_queue),
                {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
       }
#endif
       default: {
             return 0;
       }
    }
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    fInit
 * Signature: (JJ[FI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_fInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jfloatArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable float init \n");
    jfloat *fData = env->GetFloatArrayElements(cData, NULL);
    homogen_table *h_table ;
    homogenPtr tablePtr ;
    switch(getComputeDevice(cComputeDevice)) {
         case compute_device::host:{
             h_table = new homogen_table(
                                 fData, cRowCount, cColCount, detail::empty_delete<const float>(),
                                 getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
#ifdef CPU_GPU_PROFILE
         case compute_device::cpu:{
             sycl::queue cpu_queue = getQueue(compute_device::cpu);
             auto cpu_data = malloc_shared<float>(cRowCount * cColCount, cpu_queue);
             cpu_queue.memcpy(cpu_data, fData, sizeof(float) * cRowCount * cColCount).wait();
             h_table = new homogen_table(cpu_queue,
                 cpu_data, cRowCount, cColCount, detail::make_default_delete<const float>(cpu_queue),
                 {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
         case compute_device::gpu:{
             sycl::queue gpu_queue = getQueue(compute_device::gpu);
             auto gpu_data = malloc_shared<float>(cRowCount * cColCount, gpu_queue);
             gpu_queue.memcpy(gpu_data, fData, sizeof(float) * cRowCount * cColCount).wait();
             h_table = new homogen_table(gpu_queue,
                  gpu_data, cRowCount, cColCount, detail::make_default_delete<const float>(gpu_queue),
                  {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
#endif
         default: {
             return 0;
         }
    }
}
/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    dInit
 * Signature: (JJ[DI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_dInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jdoubleArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable double init \n");
    jdouble *fData = env->GetDoubleArrayElements(cData, NULL);
    homogen_table *h_table ;
    homogenPtr tablePtr ;
    switch(getComputeDevice(cComputeDevice)) {
         case compute_device::host:{
             h_table = new homogen_table(
                             fData, cRowCount, cColCount, detail::empty_delete<const double>(),
                             getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
#ifdef CPU_GPU_PROFILE
         case compute_device::cpu:{
             sycl::queue cpu_queue = getQueue(compute_device::cpu);
             auto cpu_data = malloc_shared<double>(cRowCount * cColCount, cpu_queue);
             cpu_queue.memcpy(cpu_data, fData, sizeof(double) * cRowCount * cColCount).wait();
             h_table = new homogen_table(cpu_queue,
                 cpu_data, cRowCount, cColCount, detail::make_default_delete<const double>(cpu_queue),
                 {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
         case compute_device::gpu:{
             sycl::queue gpu_queue = getQueue(compute_device::gpu);
             auto gpu_data = malloc_shared<double>(cRowCount * cColCount, gpu_queue);
             gpu_queue.memcpy(gpu_data, fData, sizeof(double) * cRowCount * cColCount).wait();
             h_table = new homogen_table(gpu_queue,
                  gpu_data, cRowCount, cColCount, detail::make_default_delete<const double>(gpu_queue),
                  {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
#endif
         default: {
             return 0;
         }
    }
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    lInit
 * Signature: (JJ[JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_lInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jlongArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable long init \n");
    jlong *fData = env->GetLongArrayElements(cData, NULL);
    homogen_table *h_table ;
    homogenPtr tablePtr ;
    switch(getComputeDevice(cComputeDevice)) {
         case compute_device::host:{
             h_table = new homogen_table(
                fData, cRowCount, cColCount, detail::empty_delete<const long>(),
                getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
#ifdef CPU_GPU_PROFILE
         case compute_device::cpu:{
             sycl::queue cpu_queue = getQueue(compute_device::cpu);
             auto cpu_data = malloc_shared<long>(cRowCount * cColCount, cpu_queue);
             cpu_queue.memcpy(cpu_data, fData, sizeof(long) * cRowCount * cColCount).wait();
             h_table = new homogen_table(cpu_queue,
                 cpu_data, cRowCount, cColCount, detail::make_default_delete<const long>(cpu_queue),
                 {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
         case compute_device::gpu:{
             sycl::queue gpu_queue = getQueue(compute_device::gpu);
             auto gpu_data = malloc_shared<long>(cRowCount * cColCount, gpu_queue);
             gpu_queue.memcpy(gpu_data, fData, sizeof(long) * cRowCount * cColCount).wait();
             h_table = new homogen_table(gpu_queue,
                  gpu_data, cRowCount, cColCount, detail::make_default_delete<const long>(gpu_queue),
                  {}, getDataLayout(cLayout));
             tablePtr = std::make_shared<homogen_table>(*h_table);
             saveShareHomogenPtrVector(tablePtr);
             return (jlong)tablePtr.get();
         }
#endif
         default: {
             return 0;
         }
    }
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetColumnCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetColumnCount(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getcolumncount %ld \n", cTableAddr);
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    return htable.get_column_count();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetRowCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetRowCount(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getrowcount \n");
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    return htable.get_row_count();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetKind
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetKind(JNIEnv *env, jobject,
                                                          jlong cTableAddr) {
    printf("HomogenTable getkind \n");
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    return htable.get_kind();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetDataLayout
 * Signature: ()J
 */
JNIEXPORT jint JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetDataLayout(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getDataLayout \n");
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    return (jint)htable.get_data_layout();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetMetaData
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetMetaData(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getMetaData \n");
    homogen_table htable = *((homogen_table *)cTableAddr);
    table_metadata *mdata = (table_metadata *)&(htable.get_metadata());
    metadataPtr *metaPtr = new metadataPtr(mdata);
    saveShareMetaPtrVector(*metaPtr);
    return (jlong)metaPtr->get();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetData
 * Signature: ()J
 */
JNIEXPORT jintArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetIntData(JNIEnv *env,
                                                             jobject,
                                                             jlong cTableAddr) {
    printf("HomogenTable getIntData \n");
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    const int *data = htable.get_data<int>();
    const int datasize = htable.get_column_count() * htable.get_row_count();

    jintArray newIntArray = env->NewIntArray(datasize);
    env->SetIntArrayRegion(newIntArray, 0, datasize, data);
    return newIntArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetFloatData
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetFloatData(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getFloatData \n");
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    const float *data = htable.get_data<float>();
    const int datasize = htable.get_column_count() * htable.get_row_count();

    jfloatArray newFloatArray = env->NewFloatArray(datasize);
    env->SetFloatArrayRegion(newFloatArray, 0, datasize, data);
    return newFloatArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetLongData
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetLongData(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getLongData \n");
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    const long *data = htable.get_data<long>();
    const int datasize = htable.get_column_count() * htable.get_row_count();

    jlongArray newLongArray = env->NewLongArray(datasize);
    env->SetLongArrayRegion(newLongArray, 0, datasize, data);
    return newLongArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetDoubleData
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetDoubleData(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getDoubleData \n");
    homogen_table htable =
        *((homogen_table *)cTableAddr);
    const double *data = htable.get_data<double>();
    const int datasize = htable.get_column_count() * htable.get_row_count();

    jdoubleArray newDoubleArray = env->NewDoubleArray(datasize);
    env->SetDoubleArrayRegion(newDoubleArray, 0, datasize, data);
    return newDoubleArray;
}
