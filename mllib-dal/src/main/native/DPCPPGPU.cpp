#include <list>
#include <memory>
#include <unistd.h>

#include "DPCPPGPU.h"

typedef std::shared_ptr<sycl::queue> queuePtr;

static std::mutex mtx;
static std::vector<sycl::queue> cVector;

static sycl::queue &getSyclQueue(const sycl::device device) {
    mtx.lock();
    if (!cVector.empty()) {
        mtx.unlock();
        return cVector[0];
    } else {
        sycl::queue queue{device};
        cVector.push_back(queue);
        mtx.unlock();
        return cVector[0];
    }
}

sycl::queue &getQueue(const compute_device device) {
    std::cout << "Get Queue" << std::endl;

    switch (device) {
    case compute_device::gpu: {
        std::cout << "selector GPU" << std::endl;
        auto device_gpu = sycl::gpu_selector{}.select_device();
        return getSyclQueue(device_gpu);
    }
    case compute_device::cpu: {
        std::cout << "selector CPU" << std::endl;
        auto device_cpu = sycl::cpu_selector{}.select_device();
        return getSyclQueue(device_cpu);
    }
    default: {
        std::cout << "No Device!" << std::endl;
        exit(-1);
    }
    }
}
