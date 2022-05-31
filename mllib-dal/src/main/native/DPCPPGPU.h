#pragma once

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "service.h"
#include <CL/sycl.hpp>
#include <jni.h>

sycl::queue getQueue(const compute_device device);
