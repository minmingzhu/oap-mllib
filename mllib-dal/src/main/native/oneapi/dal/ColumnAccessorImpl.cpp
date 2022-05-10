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

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#endif
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "com_intel_oneapi_dal_table_ColumnAccessor.h"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

/*
 * Class:     com_intel_oneapi_dal_table_ColumnAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_ColumnAccessor_cPullDouble
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cColumnIndex, jlong cRowStartIndex,
   jlong cRowEndIndex, jint cComputeDevice) {
  printf("ColumnAccessor PullDouble \n");
  homogen_table *htable =
          ((std::shared_ptr<homogen_table> *)cTableAddr)->get();
  column_accessor<const double> *acc = new column_accessor<const double>(*htable);
  jdoubleArray newDoubleArray;
  switch(getComputeDevice(cComputeDevice)) {
       case compute_device::host:{
              const auto col_values = acc->pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
              newDoubleArray = env->NewDoubleArray(col_values.get_count());
              env->SetDoubleArrayRegion(newDoubleArray, 0, col_values.get_count(), col_values.get_data());
              return newDoubleArray;
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:{
              sycl::queue *cpu_queue = getQueue(compute_device::cpu);
              const auto cpu_col_values = acc->pull(*cpu_queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
              newDoubleArray = env->NewDoubleArray(cpu_col_values.get_count());
              env->SetDoubleArrayRegion(newDoubleArray, 0, cpu_col_values.get_count(), cpu_col_values.get_data());
              return newDoubleArray;
       }
       case compute_device::gpu:{
              sycl::queue *gpu_queue = getQueue(compute_device::gpu);
              const auto gpu_col_values = acc->pull(*gpu_queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
              newDoubleArray = env->NewDoubleArray(gpu_col_values.get_count());
              env->SetDoubleArrayRegion(newDoubleArray, 0, gpu_col_values.get_count(), gpu_col_values.get_data());
              return newDoubleArray;
       }
#endif
       default: {
             return newDoubleArray;
       }
    }
  }

/*
 * Class:     com_intel_oneapi_dal_table_ColumnAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_oneapi_dal_table_ColumnAccessor_cPullFloat
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cColumnIndex, jlong cRowStartIndex,
   jlong cRowEndIndex, jint cComputeDevice) {
  printf("ColumnAccessor PullFloat \n");
  homogen_table *htable =
          ((std::shared_ptr<homogen_table> *)cTableAddr)->get();
  column_accessor<const float> *acc = new column_accessor<const float>(*htable);
  jfloatArray newFloatArray;
  switch(getComputeDevice(cComputeDevice)) {
       case compute_device::host:{
              const auto col_values = acc->pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
              newFloatArray = env->NewFloatArray(col_values.get_count());
              env->SetFloatArrayRegion(newFloatArray, 0, col_values.get_count(), col_values.get_data());
              return newFloatArray;
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:{
              sycl::queue *cpu_queue = getQueue(compute_device::cpu);
              const auto cpu_col_values = acc->pull(*cpu_queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
              newFloatArray = env->NewFloatArray(cpu_col_values.get_count());
              env->SetFloatArrayRegion(newFloatArray, 0, cpu_col_values.get_count(), cpu_col_values.get_data());
              return newFloatArray;
       }
       case compute_device::gpu:{
              sycl::queue *gpu_queue = getQueue(compute_device::gpu);
              const auto gpu_col_values = acc->pull(*gpu_queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
              newFloatArray = env->NewFloatArray(gpu_col_values.get_count());
              env->SetFloatArrayRegion(newFloatArray, 0, gpu_col_values.get_count(), gpu_col_values.get_data());
              return newFloatArray;
       }
#endif
       default: {
             return newFloatArray;
       }
    }
  }

/*
* Class:     com_intel_oneapi_dal_table_ColumnAccessor
* Method:    cPull
* Signature: (JJ)[D
*/
JNIEXPORT jintArray JNICALL Java_com_intel_oneapi_dal_table_ColumnAccessor_cPullInt
(JNIEnv *env, jobject, jlong cTableAddr, jlong cColumnIndex, jlong cRowStartIndex,
 jlong cRowEndIndex, jint cComputeDevice) {
printf("ColumnAccessor PullInt \n");
homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)cTableAddr)->get();
column_accessor<const int> *acc = new column_accessor<const int>(*htable);
jintArray newIntArray;
switch(getComputeDevice(cComputeDevice)) {
     case compute_device::host:{
            const auto col_values = acc->pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
            newIntArray = env->NewIntArray(col_values.get_count());
            env->SetIntArrayRegion(newIntArray, 0, col_values.get_count(), col_values.get_data());
            return newIntArray;
     }
#ifdef CPU_GPU_PROFILE
     case compute_device::cpu:{
            sycl::queue *cpu_queue = getQueue(compute_device::cpu);
            const auto cpu_col_values = acc->pull(*cpu_queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
            newIntArray = env->NewIntArray(cpu_col_values.get_count());
            env->SetIntArrayRegion(newIntArray, 0, cpu_col_values.get_count(), cpu_col_values.get_data());
            return newIntArray;
     }
     case compute_device::gpu:{
            sycl::queue *gpu_queue = getQueue(compute_device::gpu);
            const auto gpu_col_values = acc->pull(*gpu_queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
            newIntArray = env->NewIntArray(gpu_col_values.get_count());
            env->SetIntArrayRegion(newIntArray, 0, gpu_col_values.get_count(), gpu_col_values.get_data());
            return newIntArray;
     }
#endif
     default: {
           return newIntArray;
     }
  }
}
