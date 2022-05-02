#pragma once

#include "service.h"
#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <daal_sycl.h>
#include <jni.h>
#include <oneapi/ccl.hpp>
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"


sycl::device getAssignedGPU(ccl::communicator &comm, int size, int rankId,
                            jint *gpu_indices, int n_gpu);

sycl::queue getQueue(const compute_device device);
