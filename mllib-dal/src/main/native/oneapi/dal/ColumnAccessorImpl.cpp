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

using namespace std;
using namespace oneapi::dal;

/*
 * Class:     com_intel_oneapi_dal_table_ColumnAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_ColumnAccessor_cPullDouble
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cColumnIndex, jlong cRowStartIndex, jlong cRowEndIndex) {
      homogen_table *htable =
              ((std::shared_ptr<homogen_table> *)cTableAddr)->get();
      column_accessor<const double> *acc = new column_accessor<const double>(*htable);
      #ifdef CPU_GPU_PROFILE
          sycl::queue queue = getQueue();
          const auto sycl_col_values = acc->pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
          double* sycl_col_doubles = new double[sycl_col_values.get_count()];
          for (std::int64_t i = 0; i < sycl_col_values.get_count(); i++) {
              sycl_col_doubles[i] = sycl_col_values[i];
          }
          jdoubleArray sycl_newDoubleArray = env->NewDoubleArray(sycl_col_values.get_count());
          env->SetDoubleArrayRegion(sycl_newDoubleArray, 0, sycl_col_values.get_count(), sycl_col_doubles);
          return sycl_newDoubleArray;
      #endif
          const auto col_values = acc->pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
          double* col_doubles = new double[col_values.get_count()];
          for (std::int64_t i = 0; i < col_values.get_count(); i++) {
              col_doubles[i] = col_values[i];
          }
          jdoubleArray newDoubleArray = env->NewDoubleArray(col_values.get_count());
          env->SetDoubleArrayRegion(newDoubleArray, 0, col_values.get_count(), col_doubles);
          return newDoubleArray;
  }
