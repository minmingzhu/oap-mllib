#pragma once

#include <CL/sycl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

sycl::queue *getQueue(const bool is_gpu);
