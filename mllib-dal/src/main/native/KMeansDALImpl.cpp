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

#include <chrono>
#include <iostream>
#include <iomanip>


#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#endif

#include "OneCCL.h"
#include "com_intel_oap_mllib_clustering_KMeansDALImpl.h"
#include "service.h"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"
#include "oneapi/dal/table/homogen.hpp"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

typedef double algorithmFPType; /* Algorithm floating-point type */
typedef std::shared_ptr<oneapi::dal::homogen_table> HomogenTablePtr;

static NumericTablePtr kmeans_compute(int rankId, ccl::communicator &comm,
                                      const NumericTablePtr &pData,
                                      const NumericTablePtr &initialCentroids,
                                      size_t nClusters, size_t nBlocks,
                                      algorithmFPType &ret_cost) {
    const bool isRoot = (rankId == ccl_root);
    size_t CentroidsArchLength = 0;
    InputDataArchive inputArch;
    if (isRoot) {
        /* Retrieve the algorithm results and serialize them */
        initialCentroids->serialize(inputArch);
        CentroidsArchLength = inputArch.getSizeOfArchive();
    }

    /* Get partial results from the root node */
    ccl::broadcast(&CentroidsArchLength, sizeof(size_t), ccl::datatype::uint8,
                   ccl_root, comm)
        .wait();

    ByteBuffer nodeCentroids(CentroidsArchLength);
    if (isRoot)
        inputArch.copyArchiveToArray(&nodeCentroids[0], CentroidsArchLength);

    ccl::broadcast(&nodeCentroids[0], CentroidsArchLength, ccl::datatype::uint8,
                   ccl_root, comm)
        .wait();

    /* Deserialize centroids data */
    OutputDataArchive outArch(nodeCentroids.size() ? &nodeCentroids[0] : NULL,
                              CentroidsArchLength);

    NumericTablePtr centroids(new HomogenNumericTable<algorithmFPType>());

    centroids->deserialize(outArch);

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::Distributed<step1Local, algorithmFPType> localAlgorithm(nClusters);

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(kmeans::data, pData);
    localAlgorithm.input.set(kmeans::inputCentroids, centroids);

    /* Compute k-means */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    ByteBuffer serializedData;

    /* Serialized data is of equal size on each node if each node called
     * compute() equal number of times */
    vector<size_t> recvCounts(nBlocks);
    for (size_t i = 0; i < nBlocks; i++) {
        recvCounts[i] = perNodeArchLength;
    }
    serializedData.resize(perNodeArchLength * nBlocks);

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    ccl::allgatherv(&nodeResults[0], perNodeArchLength, &serializedData[0],
                    recvCounts, ccl::datatype::uint8, comm)
        .wait();

    if (isRoot) {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::Distributed<step2Master, algorithmFPType> masterAlgorithm(
            nClusters);

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[perNodeArchLength * i],
                                       perNodeArchLength);

            kmeans::PartialResultPtr dataForStep2FromStep1(
                new kmeans::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm
             */
            masterAlgorithm.input.add(kmeans::partialResults,
                                      dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        ret_cost = masterAlgorithm.getResult()
                       ->get(kmeans::objectiveFunction)
                       ->getValue<algorithmFPType>(0, 0);

        /* Retrieve the algorithm results */
        return masterAlgorithm.getResult()->get(kmeans::centroids);
    }
    return NumericTablePtr();
}

static bool isCenterConverged(const algorithmFPType *oldCenter,
                              const algorithmFPType *newCenter, size_t dim,
                              double tolerance) {

    algorithmFPType sums = 0.0;

    for (size_t i = 0; i < dim; i++)
        sums += (newCenter[i] - oldCenter[i]) * (newCenter[i] - oldCenter[i]);

    return sums <= tolerance * tolerance;
}

static bool areAllCentersConverged(const NumericTablePtr &oldCenters,
                                   const NumericTablePtr &newCenters,
                                   double tolerance) {
    size_t rows = oldCenters->getNumberOfRows();
    size_t cols = oldCenters->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> blockOldCenters;
    oldCenters->getBlockOfRows(0, rows, readOnly, blockOldCenters);
    algorithmFPType *arrayOldCenters = blockOldCenters.getBlockPtr();

    BlockDescriptor<algorithmFPType> blockNewCenters;
    newCenters->getBlockOfRows(0, rows, readOnly, blockNewCenters);
    algorithmFPType *arrayNewCenters = blockNewCenters.getBlockPtr();

    for (size_t i = 0; i < rows; i++) {
        if (!isCenterConverged(&arrayOldCenters[i * cols],
                               &arrayNewCenters[i * cols], cols, tolerance))
            return false;
    }

    return true;
}

static jlong doKMeansDALComputeWithInitCenters(
    JNIEnv *env, jobject obj, int rankId, ccl::communicator &comm,
    NumericTablePtr &pData, NumericTablePtr &centroids, jint cluster_num,
    jdouble tolerance, jint iteration_num, jint executor_num,
    jobject resultObj) {

    algorithmFPType totalCost;

    NumericTablePtr newCentroids;
    bool converged = false;

    int it = 0;
    for (it = 0; it < iteration_num && !converged; it++) {
        auto t1 = std::chrono::high_resolution_clock::now();

        newCentroids = kmeans_compute(rankId, comm, pData, centroids,
                                      cluster_num, executor_num, totalCost);

        if (rankId == ccl_root) {
            converged =
                areAllCentersConverged(centroids, newCentroids, tolerance);
        }

        // Sync converged status
        ccl::broadcast(&converged, 1, ccl::datatype::uint8, ccl_root, comm)
            .wait();

        centroids = newCentroids;

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        std::cout << "KMeans (native): iteration " << it << " took " << duration
                  << " secs" << std::endl;
    }

    if (rankId == ccl_root) {
        if (it == iteration_num)
            std::cout << "KMeans (native): reached " << iteration_num
                      << " max iterations." << std::endl;
        else
            std::cout << "KMeans (native): converged in " << it
                      << " iterations." << std::endl;

        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID totalCostField = env->GetFieldID(clazz, "totalCost", "D");
        jfieldID iterationNumField =
            env->GetFieldID(clazz, "iterationNum", "I");

        // Set iteration num for result
        env->SetIntField(resultObj, iterationNumField, it);
        // Set cost for result
        env->SetDoubleField(resultObj, totalCostField, totalCost);

        NumericTablePtr *ret = new NumericTablePtr(centroids);
        return (jlong)ret;
    } else
        return (jlong)0;
}

static jlong doKMeansOneAPICompute(
  JNIEnv *env, jobject obj, int rankId, ccl::communicator &comm,
  jlong pNumTabData, jlong pNumTabCenters, jint cluster_num,
  jdouble tolerance, jint iteration_num, jobject resultObj, jint cComputeDevice) {
       oneapi::dal::homogen_table *htable =
          ((HomogenTablePtr *)pNumTabData)->get();
       oneapi::dal::homogen_table *centroids =
          ((HomogenTablePtr *)pNumTabCenters)->get();
       const auto kmeans_desc = oneapi::dal::kmeans::descriptor<>()
                                           .set_cluster_count(cluster_num)
                                           .set_max_iteration_count(iteration_num)
                                           .set_accuracy_threshold(tolerance);
       extern const std::int64_t rank_id;
       extern const auto result_train;
switch(getComputeDevice(cComputeDevice)) {
       case compute_device::host:{
          const auto x_train = oneapi::dal::read<oneapi::dal::homogen_table>(*htable);
          const auto initial_centroids = oneapi::dal::read<oneapi::dal::homogen_table>(*centroids);
          auto comm = oneapi::dal::preview::spmd::make_communicator<oneapi::dal::preview::spmd::backend::ccl>();
          rank_id = comm.get_rank();
//          auto rank_count = comm.get_rank_count();
//          auto input_vec = split_table_by_rows<double>(queue, x_train, rank_count);
//          dal::kmeans::train_input local_input { input_vec[rank_id], initial_centroids };
          result_train = oneapi::dal::preview::train(comm, kmeans_desc, x_train, initial_centroids)
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:{
          sycl::queue *cpu_queue = getQueue(false);
          const auto x_train = oneapi::dal::read<oneapi::dal::homogen_table>(*cpu_queue, *htable);
          const auto initial_centroids = oneapi::dal::read<oneapi::dal::homogen_table>(*cpu_queue, *centroids);
          auto comm = oneapi::dal::preview::spmd::make_communicator<oneapi::dal::preview::spmd::backend::ccl>(*cpu_queue);
          rank_id = comm.get_rank();
          auto rank_count = comm.get_rank_count();
          auto input_vec = split_table_by_rows<double>(*cpu_queue, x_train, rank_count);
          oneapi::dal::kmeans::train_input local_input { input_vec[rank_id], initial_centroids };
          result_train = oneapi::dal::preview::train(comm, kmeans_desc, local_input)
       }
       case compute_device::gpu:{
          sycl::queue *gpu_queue = getQueue(true);
          const auto x_train = oneapi::dal::read<oneapi::dal::homogen_table>(*gpu_queue, *htable);
          const auto initial_centroids = oneapi::dal::read<oneapi::dal::homogen_table>(*gpu_queue, *centroids);
          auto comm = oneapi::dal::preview::spmd::make_communicator<oneapi::dal::preview::spmd::backend::ccl>(*gpu_queue);
          rank_id = comm.get_rank();
          auto rank_count = comm.get_rank_count();
          auto input_vec = split_table_by_rows<double>(*gpu_queue, x_train, rank_count);
          oneapi::dal::kmeans::train_input local_input { input_vec[rank_id], initial_centroids };
          result_train = oneapi::dal::preview::train(comm, kmeans_desc, local_input)
       }
#endif
       default: {
             return (jlong)0;
       }
    }

       if(rank_id == 0) {
            std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
            std::cout << "Objective function value: " << result_train.get_objective_function_value()
                      << std::endl;
            std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
               // Get the class of the input object
            jclass clazz = env->GetObjectClass(resultObj);
            // Get Field references
            jfieldID totalCostField = env->GetFieldID(clazz, "totalCost", "D");
            jfieldID iterationNumField =
                env->GetFieldID(clazz, "iterationNum", "I");
            // Set iteration num for result
            env->SetIntField(resultObj, iterationNumField, result_train.get_iteration_count());
            // Set cost for result
            env->SetDoubleField(resultObj, totalCostField, result_train.get_objective_function_value());

            oneapi::dal::homogen_table *centroidsTable = new oneapi::dal::homogen_table(result_train.get_model().get_centroids());
            HomogenTablePtr centroidsPtr = new HomogenTablePtr(centroidsTable);
            return (jlong)centroidsPtr;
       } else {
            return (jlong)0;
       }
}


JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_clustering_KMeansDALImpl_cKMeansDALComputeWithInitCenters(
    JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabCenters,
    jint cluster_num, jdouble tolerance, jint iteration_num, jint executor_num,
    jint executor_cores, jboolean use_gpu, jintArray gpu_idx_array,
    jobject resultObj) {

    ccl::communicator &comm = getComm();
    int rankId = comm.rank();

    NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
    NumericTablePtr centroids = *((NumericTablePtr *)pNumTabCenters);

    jlong ret = 0L;
#ifdef CPU_GPU_PROFILE

    if (use_gpu) {
        cout << "oneDAL (native): use GPU kernels with " << n_gpu << " GPU(s)"
             << endl;

        ret = doKMeansDALComputeWithInitCenters(
            env, obj, rankId, comm, pNumTabData, centroids, cluster_num,
            tolerance, iteration_num, executor_num, resultObj, cComputeDevice);
    } else
#endif
    {
        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(
            executor_cores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        cout << "oneDAL (native): use CPU kernels with " << nThreadsNew
             << " threads" << endl;
        ret = doKMeansDALComputeWithInitCenters(
            env, obj, rankId, comm, pData, centroids, cluster_num, tolerance,
            iteration_num, executor_num, resultObj);
    }

    return ret;
}
