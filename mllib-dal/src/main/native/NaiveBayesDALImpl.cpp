#include <unistd.h>

#include "Logger.h"
#include "OneCCL.h"
#include "com_intel_oap_mllib_classification_NaiveBayesDALImpl.h"
#include "service.h"

#define PROFILE 1

#ifdef PROFILE
#include "Profile.hpp"
#endif

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::multinomial_naive_bayes;

template <training::Method method>
static training::ResultPtr
trainModel(const ccl::communicator &comm, const NumericTablePtr &featuresTab,
           const NumericTablePtr &labelsTab, int nClasses) {
#ifdef PROFILE
    Profiler profiler("NaiveBayes");
#endif

    size_t rankId = comm.rank();
    size_t nBlocks = comm.size();

    /* Create an algorithm object to train the Naive Bayes model based on the
     * local-node data */
    training::Distributed<step1Local, CpuAlgorithmFPType, method>
        localAlgorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(classifier::training::data, featuresTab);
    localAlgorithm.input.set(classifier::training::labels, labelsTab);

#ifdef PROFILE
    profiler.startProfile("local step compute");
#endif

    /* Train the Naive Bayes model on local nodes */
    localAlgorithm.compute();

#ifdef PROFILE
    profiler.endProfile();
#endif

    /* Serialize partial results required by step 2 */
    services::SharedPtr<daal::byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called
     * compute() equal number of times */
    if (rankId == ccl_root) {
        serializedData.reset(new daal::byte[perNodeArchLength * nBlocks]);
    }

    {
        services::SharedPtr<daal::byte> nodeResults(
            new daal::byte[perNodeArchLength]);
        dataArch.copyArchiveToArray(nodeResults.get(), perNodeArchLength);

#ifdef PROFILE
        profiler.startProfile("ccl::gather");
#endif

        /* Transfer partial results to step 2 on the root node */
        ccl::gather(nodeResults.get(), perNodeArchLength, serializedData.get(),
                    perNodeArchLength, comm)
            .wait();

#ifdef PROFILE
        profiler.endProfile();
#endif
    }

    if (rankId == ccl_root) {
        /* Create an algorithm object to build the final Naive Bayes model on
         * the master node */
        training::Distributed<step2Master, CpuAlgorithmFPType, method>
            masterAlgorithm(nClasses);

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            training::PartialResultPtr dataForStep2FromStep1(
                new training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local Naive Bayes model as input for the master-node
             * algorithm */
            masterAlgorithm.input.add(training::partialModels,
                                      dataForStep2FromStep1);
        }

#ifdef PROFILE
        profiler.startProfile("master step compute");
#endif
        /* Merge and finalizeCompute the Naive Bayes model on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

#ifdef PROFILE
        profiler.endProfile();
#endif

        /* Retrieve the algorithm results */
        training::ResultPtr trainingResult = masterAlgorithm.getResult();

        return trainingResult;
    }
    return training::ResultPtr();
}

JNIEXPORT void JNICALL
Java_com_intel_oap_mllib_classification_NaiveBayesDALImpl_cNaiveBayesDALCompute(
    JNIEnv *env, jobject obj, jlong pFeaturesTab, jlong pLabelsTab,
    jint class_num, jint executor_num, jint executor_cores, jobject resultObj) {

    ccl::communicator &comm = getComm();
    size_t rankId = comm.rank();

    NumericTablePtr featuresTab = *((NumericTablePtr *)pFeaturesTab);
    NumericTablePtr labelsTab = *((NumericTablePtr *)pLabelsTab);

    // Set number of threads for OneDAL to use for each rank
    services::Environment::getInstance()->setNumberOfThreads(executor_cores);

    int nThreadsNew =
        services::Environment::getInstance()->getNumberOfThreads();
    logger::println(logger::INFO,
                    "OneDAL (native): Number of CPU threads used %d",
                    nThreadsNew);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Support both dense and csr numeric table
    training::ResultPtr trainingResult;
    if (featuresTab->getDataLayout() == NumericTable::StorageLayout::csrArray) {
        logger::println(logger::INFO,
                        "OneDAL (native): training model with fastCSR method");
        trainingResult = trainModel<training::fastCSR>(comm, featuresTab,
                                                       labelsTab, class_num);
    } else {
        logger::println(
            logger::INFO,
            "OneDAL (native): training model with defaultDense method");
        trainingResult = trainModel<training::defaultDense>(
            comm, featuresTab, labelsTab, class_num);
    }

    logger::println(logger::INFO, "OneDAL (native): training model finished");

    if (rankId == ccl_root) {
        multinomial_naive_bayes::ModelPtr model =
            trainingResult->get(classifier::training::model);
        auto t2 = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration<float>(t2 - t1).count();
        logger::println(logger::INFO,
                        "NativeBayes (native): training step took  %f secs",
                        duration);
        // Return all log of class priors (LogP) and log of class conditional
        // probabilities (LogTheta)

        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID piNumericTableField =
            env->GetFieldID(clazz, "piNumericTable", "J");
        jfieldID thetaNumericTableField =
            env->GetFieldID(clazz, "thetaNumericTable", "J");

        NumericTablePtr *pi = new NumericTablePtr(model->getLogP());
        NumericTablePtr *theta = new NumericTablePtr(model->getLogTheta());

        env->SetLongField(resultObj, piNumericTableField, (jlong)pi);
        env->SetLongField(resultObj, thetaNumericTableField, (jlong)theta);
    }
}
