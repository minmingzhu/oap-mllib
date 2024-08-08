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
#include "Logger.h"
#include "Common.hpp"

#include "com_intel_oneapi_dal_table_RowAccessor.h"
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
   jint computeDeviceOrdinal){
   try {
          logger::println(logger::INFO, "RowAccessor PullDouble");
          logger::println(logger::INFO, "RowAccessor cRowStartIndex %d", cRowStartIndex);
          logger::println(logger::INFO, "RowAccessor cRowEndIndex %d", cRowEndIndex);

          homogen_table htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
          row_accessor<const double> acc {htable};
          jdoubleArray newDoubleArray = nullptr;
          oneapi::dal::array<double> row_values;
          ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
          switch(device) {
                 case ComputeDevice::host:{
                        row_values = acc.pull({cRowStartIndex, cRowEndIndex});
                        break;
                 }
                 case ComputeDevice::cpu:
                 case ComputeDevice::gpu:{
                        auto queue = getQueue(device);
                        row_values = acc.pull(queue, {cRowStartIndex, cRowEndIndex});
                        break;
                 }
                 default: {
                       return newDoubleArray;
                 }
              }
              logger::println(logger::INFO, "RowAccessor get_count %d", row_values.get_count());
              newDoubleArray = env->NewDoubleArray(row_values.get_count());
              env->SetDoubleArrayRegion(newDoubleArray, 0, row_values.get_count(),  row_values.get_data());
              // Get the length of the jdoubleArray
              jsize length = env->GetArrayLength(newDoubleArray);
              logger::println(logger::INFO, "newDoubleArray size %d", length);
        //      std::cout << "Values in the jdoubleArray: ";
        //
        //      jboolean isCopy;
        //      jdouble* elements = env->GetDoubleArrayElements(newDoubleArray, &isCopy);
        //
        //      // Use the elements
        //      for (int i = 0; i < row_values.get_count(); ++i) {
        //            std::cout << elements[i] << " ";
        //      }
        //
        //      // Release the elements
        //      env->ReleaseDoubleArrayElements(newDoubleArray, elements, 0);
              logger::println(logger::INFO, "return newDoubleArray");
              return newDoubleArray;
    } catch (const std::exception& e) {
        // Handle exception
        std::cerr << "Exception occurred while pulling data: " << e.what() << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown exception occurred while pulling data." << std::endl;
    }

  }

/*
 * Class:     com_intel_oneapi_dal_table_RowAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_oneapi_dal_table_RowAccessor_cPullFloat
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cRowStartIndex, jlong cRowEndIndex,
   jint computeDeviceOrdinal){
  logger::println(logger::INFO, "RowAccessor PullFloat");
  homogen_table htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
  row_accessor<const float> acc { htable };
  jfloatArray newFloatArray = nullptr;
  oneapi::dal::array<float> row_values;
  ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
  switch(device) {
         case ComputeDevice::host:{
                row_values = acc.pull({cRowStartIndex, cRowEndIndex});
                break;
         }
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
                auto queue = getQueue(device);
                row_values = acc.pull(queue, {cRowStartIndex, cRowEndIndex});
                break;
         }
         default: {
               return newFloatArray;
         }
      }
      newFloatArray = env->NewFloatArray(row_values.get_count());
      env->SetFloatArrayRegion(newFloatArray, 0, row_values.get_count(), row_values.get_data());
      return newFloatArray;
  }

/*
 * Class:     com_intel_oneapi_dal_table_RowAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jintArray JNICALL Java_com_intel_oneapi_dal_table_RowAccessor_cPullInt
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cRowStartIndex, jlong cRowEndIndex,
   jint computeDeviceOrdinal){
  logger::println(logger::INFO, "RowAccessor PullInt");
  homogen_table htable = *reinterpret_cast<homogen_table *>(cTableAddr);
  row_accessor<const int> acc { htable };
  jintArray newIntArray = nullptr;
  oneapi::dal::array<int> row_values;
  ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
  switch(device) {
         case ComputeDevice::host:{
                row_values = acc.pull({cRowStartIndex, cRowEndIndex});
                break;
         }
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
                auto queue = getQueue(device);
                row_values = acc.pull(queue, {cRowStartIndex, cRowEndIndex});
                break;
         }
         default: {
               return newIntArray;
         }
      }
      newIntArray = env->NewIntArray(row_values.get_count());
      env->SetIntArrayRegion(newIntArray, 0, row_values.get_count(), row_values.get_data());
      return newIntArray;
  }
#endif
