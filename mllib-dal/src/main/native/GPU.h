#pragma once

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <daal_sycl.h>
#include <jni.h>
#include <oneapi/ccl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"


sycl::device getAssignedGPU(ccl::communicator &comm, int size, int rankId,
                            jint *gpu_indices, int n_gpu);

sycl::queue *getQueue(const bool is_gpu);

template <typename T>
std::vector<oneapi::dal::table> split_table_by_rows(sycl::queue& queue,
                                            const oneapi::dal::table& t,
                                            std::int64_t split_count);
