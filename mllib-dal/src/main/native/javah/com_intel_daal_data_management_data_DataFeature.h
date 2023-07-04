/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_daal_data_management_data_DataFeature */

#ifndef _Included_com_intel_daal_data_management_data_DataFeature
#define _Included_com_intel_daal_data_management_data_DataFeature
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    getPMMLNumType
 * Signature: ()Lcom/intel/daal/data_management/data/DataFeatureUtils/PMMLNumType;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_DataFeature_getPMMLNumType
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    getFeatureType
 * Signature: ()Lcom/intel/daal/data_management/data/DataFeatureUtils/FeatureType;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_DataFeature_getFeatureType
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    getCategoryNumber
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_DataFeature_getCategoryNumber
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    init
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_DataFeature_init
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetPMMLNumType
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetPMMLNumType
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetFeatureType
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetFeatureType
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetCategoryNumber
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetCategoryNumber
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetName
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetName
  (JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetDoubleType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetDoubleType
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetFloatType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetFloatType
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetLongType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetLongType
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSetIntType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSetIntType
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cSerializeCObject
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cSerializeCObject
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_DataFeature
 * Method:    cDeserializeCObject
 * Signature: (Ljava/nio/ByteBuffer;J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_DataFeature_cDeserializeCObject
  (JNIEnv *, jobject, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif