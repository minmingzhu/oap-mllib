/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_oap_mllib_OneCCL__ */

#ifndef _Included_com_intel_oap_mllib_OneCCL__
#define _Included_com_intel_oap_mllib_OneCCL__
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    isRoot
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_oap_mllib_OneCCL_00024_isRoot
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    rankID
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_rankID
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    setEnv
 * Signature: (Ljava/lang/String;Ljava/lang/String;Z)I
 */
JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_setEnv
  (JNIEnv *, jobject, jstring, jstring, jboolean);

/*
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    c_getAvailPort
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_c_1getAvailPort
  (JNIEnv *, jobject, jstring);

/*
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    c_init
 * Signature: (IILjava/lang/String;Lcom/intel/oap/mllib/CCLParam;)I
 */
JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_c_1init
  (JNIEnv *, jobject, jint, jint, jstring, jint, jobject);

/*
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    c_cleanup
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_intel_oap_mllib_OneCCL_00024_c_1cleanup
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
