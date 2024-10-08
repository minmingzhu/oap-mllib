#pragma once

#include "service.h"
#include <CL/cl.h>
#include <sycl/sycl.hpp>
#include <jni.h>
#include <oneapi/ccl.hpp>

sycl::queue getAssignedGPU(const ComputeDevice device, ccl::communicator &comm,
                           int size, int rankId, jint *gpu_indices, int n_gpu);

sycl::queue getQueue(const ComputeDevice device);

sycl::queue getGPU(const ComputeDevice device,jint *gpu_indices);

std::vector<sycl::device> get_gpus();
