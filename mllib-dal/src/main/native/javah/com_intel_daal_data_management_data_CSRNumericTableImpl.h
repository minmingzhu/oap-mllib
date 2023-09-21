/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_daal_data_management_data_CSRNumericTableImpl */

#ifndef _Included_com_intel_daal_data_management_data_CSRNumericTableImpl
#define _Included_com_intel_daal_data_management_data_CSRNumericTableImpl
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    initCSRNumericTable
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_initCSRNumericTable
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getIndexType
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getIndexType
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    cGetNumberOfRows
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_cGetNumberOfRows
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    cGetDataSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_cGetDataSize
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getColIndicesBuffer
 * Signature: (JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getColIndicesBuffer
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getRowOffsetsBuffer
 * Signature: (JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getRowOffsetsBuffer
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getDoubleBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getDoubleBuffer
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getFloatBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getFloatBuffer
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getLongBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getLongBuffer
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
