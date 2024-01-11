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

#include "com_intel_oap_mllib_OneDAL__.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

// Use OneDAL lib function
extern bool daal_check_is_intel_cpu();

// Define a global native array
typedef std::shared_ptr<double> NativeDoubleArrayPtr;
typedef std::shared_ptr<float> NativeFloatArrayPtr;

std::mutex g_amtx;
template <typename T> std::vector<std::shared_ptr<T>> g_NativeArrayPtrVector;

template <typename T>
static void saveArrayPtrToVector(const std::shared_ptr<T> &ptr) {
       g_amtx.lock();
       g_NativeArrayPtrVector<T>.push_back(ptr);
       g_amtx.unlock();
}

JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cAddNumericTable(
    JNIEnv *, jobject, jlong rowMergedNumericTableAddr,
    jlong numericTableAddr) {
    data_management::RowMergedNumericTablePtr pRowMergedNumericTable = (*(
        data_management::RowMergedNumericTablePtr *)rowMergedNumericTableAddr);
    data_management::NumericTablePtr pNumericTable =
        (*(data_management::NumericTablePtr *)numericTableAddr);
    pRowMergedNumericTable->addNumericTable(pNumericTable);
}

JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cSetDouble(
    JNIEnv *env, jobject, jlong numTableAddr, jint row, jint column,
    jdouble value) {
    HomogenNumericTable<double> *nt =
        static_cast<HomogenNumericTable<double> *>(
            ((SerializationIfacePtr *)numTableAddr)->get());
    (*nt)[row][column] = (double)value;
}

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    cSetDoubleBatch
 * Signature: (JI[DII)V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cSetDoubleBatch(
    JNIEnv *env, jobject, jlong numTableAddr, jint curRows, jdoubleArray batch,
    jint numRows, jint numCols) {
    HomogenNumericTable<double> *nt =
        static_cast<HomogenNumericTable<double> *>(
            ((SerializationIfacePtr *)numTableAddr)->get());
    jdouble *values = (jdouble *)env->GetPrimitiveArrayCritical(batch, 0);
    if (values == NULL) {
        logger::println(logger::INFO,
                        "Error: unable to obtain critical array.");
        exit(-1);
    }
    std::memcpy((*nt)[curRows], values, numRows * numCols * sizeof(double));
    env->ReleasePrimitiveArrayCritical(batch, values, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cFreeDataMemory(
    JNIEnv *, jobject, jlong numericTableAddr) {
    data_management::NumericTablePtr pNumericTable =
        (*(data_management::NumericTablePtr *)numericTableAddr);
    pNumericTable->freeDataMemory();
}

JNIEXPORT jboolean JNICALL
Java_com_intel_oap_mllib_OneDAL_00024_cCheckPlatformCompatibility(JNIEnv *,
                                                                  jobject) {
    // Only guarantee compatibility and performance on Intel platforms, use
    // OneDAL lib function
    std::cout << "OneDAL CheckPlatformCompatibility" << std::endl;
    return daal_check_is_intel_cpu();
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_OneDAL_00024_cNewCSRNumericTableFloat(
    JNIEnv *env, jobject, jfloatArray data, jlongArray colIndices,
    jlongArray rowOffsets, jlong nFeatures, jlong nVectors) {

    long numData = env->GetArrayLength(data);

    size_t *resultRowOffsets = NULL;
    size_t *resultColIndices = NULL;
    float *resultData = NULL;

    CSRNumericTable *numericTable = new CSRNumericTable(
        resultData, resultColIndices, resultRowOffsets, nFeatures, nVectors);
    numericTable->allocateDataMemory(numData);
    numericTable->getArrays<float>(&resultData, &resultColIndices,
                                   &resultRowOffsets);

    size_t *pRowOffsets = (size_t *)env->GetLongArrayElements(rowOffsets, 0);
    size_t *pColIndices = (size_t *)env->GetLongArrayElements(colIndices, 0);
    float *pData = env->GetFloatArrayElements(data, 0);

    for (size_t i = 0; i < (size_t)numData; ++i) {
        resultData[i] = pData[i];
        resultColIndices[i] = pColIndices[i];
    }
    for (size_t i = 0; i < (size_t)nVectors + 1; ++i) {
        resultRowOffsets[i] = pRowOffsets[i];
    }

    env->ReleaseLongArrayElements(rowOffsets, (jlong *)pRowOffsets, 0);
    env->ReleaseLongArrayElements(colIndices, (jlong *)pColIndices, 0);
    env->ReleaseFloatArrayElements(data, pData, 0);

    CSRNumericTablePtr *ret = new CSRNumericTablePtr(numericTable);

    return (jlong)ret;
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_OneDAL_00024_cNewCSRNumericTableDouble(
    JNIEnv *env, jobject, jdoubleArray data, jlongArray colIndices,
    jlongArray rowOffsets, jlong nFeatures, jlong nVectors) {

    long numData = env->GetArrayLength(data);

    size_t *resultRowOffsets = NULL;
    size_t *resultColIndices = NULL;
    double *resultData = NULL;

    CSRNumericTable *numericTable = new CSRNumericTable(
        resultData, resultColIndices, resultRowOffsets, nFeatures, nVectors);
    numericTable->allocateDataMemory(numData);
    numericTable->getArrays<double>(&resultData, &resultColIndices,
                                    &resultRowOffsets);

    size_t *pRowOffsets = (size_t *)env->GetLongArrayElements(rowOffsets, 0);
    size_t *pColIndices = (size_t *)env->GetLongArrayElements(colIndices, 0);
    double *pData = env->GetDoubleArrayElements(data, 0);

    for (size_t i = 0; i < (size_t)numData; ++i) {
        resultData[i] = pData[i];
        resultColIndices[i] = pColIndices[i];
    }
    for (size_t i = 0; i < (size_t)nVectors + 1; ++i) {
        resultRowOffsets[i] = pRowOffsets[i];
    }

    env->ReleaseLongArrayElements(rowOffsets, (jlong *)pRowOffsets, 0);
    env->ReleaseLongArrayElements(colIndices, (jlong *)pColIndices, 0);
    env->ReleaseDoubleArrayElements(data, pData, 0);

    CSRNumericTablePtr *ret = new CSRNumericTablePtr(numericTable);

    return (jlong)ret;
}

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cNewDoubleArray
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cNewDoubleArray(
    JNIEnv *env, jobject, jlong size) {
    std::cout << "create double new native array size : " << size << std::endl;
    NativeDoubleArrayPtr arrayPtr(new double[size],
                                  [](double *ptr) { delete[] ptr; });
    saveArrayPtrToVector<double>(arrayPtr);
    return (jlong)arrayPtr.get();
}

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cNewFloatArray
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cNewFloatArray(
    JNIEnv *env, jobject, jlong size) {
    std::cout << "create float new native array size : " << size << std::endl;
    NativeFloatArrayPtr arrayPtr(new float[size],
                                 [](float *ptr) { delete[] ptr; });
    saveArrayPtrToVector<float>(arrayPtr);
    return (jlong)arrayPtr.get();
}

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cCopyDoubleArrayToNative
 * Signature: (J[DJ)V
 */
JNIEXPORT void JNICALL
Java_com_intel_oap_mllib_OneDAL_00024_cCopyDoubleArrayToNative(
    JNIEnv *env, jobject, jlong nativeArrayPtr, jdoubleArray sourceArray,
    jlong index) {
    double *nativeArray = reinterpret_cast<double *>(nativeArrayPtr);
    jsize sourceLength = env->GetArrayLength(sourceArray);
    jdouble *source = static_cast<jdouble *>(
        env->GetPrimitiveArrayCritical(sourceArray, NULL));
    std::copy(source, source + sourceLength, nativeArray + index);
    env->ReleasePrimitiveArrayCritical(sourceArray, source, JNI_ABORT);
}

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cCopyFloatArrayToNative
 * Signature: (J[DJ)V
 */
JNIEXPORT void JNICALL
Java_com_intel_oap_mllib_OneDAL_00024_cCopyFloatArrayToNative(
    JNIEnv *env, jobject, jlong nativeArrayPtr, jdoubleArray sourceArray,
    jlong index) {
    float *nativeArray = reinterpret_cast<float *>(nativeArrayPtr);
    jsize sourceLength = env->GetArrayLength(sourceArray);
    jdouble *source = static_cast<jdouble *>(
        env->GetPrimitiveArrayCritical(sourceArray, NULL));
    std::transform(source, source + sourceLength, nativeArray + index, [](double d) { return static_cast<float>(d); });
    env->ReleasePrimitiveArrayCritical(sourceArray, source, JNI_ABORT);
}
