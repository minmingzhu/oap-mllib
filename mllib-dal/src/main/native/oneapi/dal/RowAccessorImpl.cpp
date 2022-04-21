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

#include "com_intel_oneapi_dal_table_RowAccessor.h"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace std;
using namespace oneapi::dal;


/*
 * Class:     com_intel_oneapi_dal_table_RowAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_RowAccessor_cPullDouble
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cRowStartIndex, jlong cRowEndIndex){
      homogen_table *htable =
              ((std::shared_ptr<homogen_table> *)cTableAddr)->get();
      row_accessor<const double> *acc = new row_accessor<const double>(*htable);
      const auto row_values = acc->pull({cRowStartIndex, cRowEndIndex});
      double* row_doubles = new double[row_values.get_count()];
      for (std::int64_t i = 0; i < row_values.get_count(); i++) {
        row_doubles[i] = row_values[i];
      }
      jdoubleArray newDoubleArray = env->NewDoubleArray(row_values.get_count());
      env->SetDoubleArrayRegion(newDoubleArray, 0, row_values.get_count(), row_doubles);
      return newDoubleArray;
  }
