#pragma once

#include "service.h"
#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <jni.h>
#include <oneapi/ccl.hpp>

sycl::queue getAssignedGPU(const ComputeDevice device, jint *gpu_indices);

sycl::queue getQueue(const ComputeDevice device);

std::vector<sycl::device> get_gpus();
