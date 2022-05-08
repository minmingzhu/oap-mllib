#pragma once

#include "service.h"
#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <daal_sycl.h>
#include <jni.h>
#include <oneapi/ccl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

sycl::device getAssignedGPU(ccl::communicator &comm, int size, int rankId,
                            jint *gpu_indices, int n_gpu);

sycl::queue getQueue(const compute_device device);

