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

#include "com_intel_oneapi_dal_table_RowAccessor.h"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;


/*
 * Class:     com_intel_oneapi_dal_table_RowAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_RowAccessor_cPullDouble
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cRowStartIndex, jlong cRowEndIndex,
   jint cComputeDevice){
  printf("RowAccessor PullDouble \n");
  homogen_table *htable =
          ((std::shared_ptr<homogen_table> *)cTableAddr)->get();
  row_accessor<const double> *acc = new row_accessor<const double>(*htable);
  double* row_doubles;
  jdoubleArray newDoubleArray;
  switch(getComputeDevice(cComputeDevice)) {
         case compute_device::host:{
                const auto row_values = acc->pull({cRowStartIndex, cRowEndIndex});
                row_doubles = new double[row_values.get_count()];
                for (std::int64_t i = 0; i < row_values.get_count(); i++) {
                  row_doubles[i] = row_values[i];
                }
                newDoubleArray = env->NewDoubleArray(row_values.get_count());
                env->SetDoubleArrayRegion(newDoubleArray, 0, row_values.get_count(), row_doubles);
                return newDoubleArray;
         }
#ifdef CPU_GPU_PROFILE
         case compute_device::cpu:{
                sycl::queue *cpu_queue = getQueue(false);
                const auto cpu_row_values = acc->pull(*cpu_queue, {cRowStartIndex, cRowEndIndex});
                row_doubles = new double[cpu_row_values.get_count()];
                for (std::int64_t i = 0; i < cpu_row_values.get_count(); i++) {
                   row_doubles[i] = cpu_row_values[i];
                }
                newDoubleArray = env->NewDoubleArray(cpu_row_values.get_count());
                env->SetDoubleArrayRegion(newDoubleArray, 0, cpu_row_values.get_count(), row_doubles);
                return newDoubleArray;
         }
         case compute_device::gpu:{
                sycl::queue *gpu_queue = getQueue(true);
                const auto gpu_row_values = acc->pull(*gpu_queue, {cRowStartIndex, cRowEndIndex});
                row_doubles = new double[gpu_row_values.get_count()];
                for (std::int64_t i = 0; i < gpu_row_values.get_count(); i++) {
                   row_doubles[i] = gpu_row_values[i];
                }
                jdoubleArray newDoubleArray = env->NewDoubleArray(gpu_row_values.get_count());
                env->SetDoubleArrayRegion(newDoubleArray, 0, gpu_row_values.get_count(), row_doubles);
                return newDoubleArray;
         }
#endif
         default: {
               return newDoubleArray;
         }
      }
  }
