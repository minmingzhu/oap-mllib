#include <list>
#include <memory>
#include <unistd.h>

#include "DPCPPGPU.h"
#include "service.h"

sycl::queue *queue;
sycl::queue *getQueue(const bool is_gpu) {
    std::cout << "Get Queue" << std::endl;
    if (is_gpu) {
        std::cout << "selector GPU" << std::endl;
        if (queue == NULL) {
            auto device_gpu = sycl::gpu_selector{}.select_device();
            queue = new sycl::queue(device_gpu);
        }
        return queue;
    } else {
        std::cout << "selector CPU" << std::endl;
        if (queue == NULL) {
            auto device_cpu = sycl::cpu_selector{}.select_device();
            queue = new sycl::queue(device_cpu);
        }
        std::cout << "selector CPU 11" << std::endl;
        return queue;
    }
}
