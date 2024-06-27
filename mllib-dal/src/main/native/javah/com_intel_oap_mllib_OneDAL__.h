/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_oap_mllib_OneDAL__ */

#ifndef _Included_com_intel_oap_mllib_OneDAL__
#define _Included_com_intel_oap_mllib_OneDAL__
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cAddNumericTable
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cAddNumericTable
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cSetDouble
 * Signature: (JIID)V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cSetDouble
  (JNIEnv *, jobject, jlong, jint, jint, jdouble);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cSetDoubleBatch
 * Signature: (JI[DII)V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cSetDoubleBatch
  (JNIEnv *, jobject, jlong, jint, jdoubleArray, jint, jint);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cFreeDataMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cFreeDataMemory
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cCheckPlatformCompatibility
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cCheckPlatformCompatibility
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cNewCSRNumericTableFloat
 * Signature: ([F[J[JJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cNewCSRNumericTableFloat
  (JNIEnv *, jobject, jfloatArray, jlongArray, jlongArray, jlong, jlong);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cNewCSRNumericTableDouble
 * Signature: ([D[J[JJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cNewCSRNumericTableDouble
  (JNIEnv *, jobject, jdoubleArray, jlongArray, jlongArray, jlong, jlong);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cNewDoubleArray
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cNewDoubleArray
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cCopyDoubleArrayToNative
 * Signature: (J[DJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cCopyDoubleArrayToNative
  (JNIEnv *, jobject, jlong, jdoubleArray, jlong);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cNewFloatArray
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cNewFloatArray
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_oap_mllib_OneDAL__
 * Method:    cCopyFloatArrayToNative
 * Signature: (J[DJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneDAL_00024_cCopyFloatArrayToNative
  (JNIEnv *, jobject, jlong, jdoubleArray, jlong);

#ifdef __cplusplus
}
#endif
#endif
