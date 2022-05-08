#include <list>
#include <memory>
#include <unistd.h>

#include "GPU.h"
#include "service.h"

sycl::queue *queue;

static std::vector<sycl::device> get_gpus() {
    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            return devices;
        }
    }
    std::cout << "No GPUs!" << std::endl;
    exit(-1);

    return {};
}

static int getLocalRank(ccl::communicator &comm, int size, int rank) {
    const int MPI_MAX_PROCESSOR_NAME = 128;
    /* Obtain local rank among nodes sharing the same host name */
    char zero = static_cast<char>(0);
    std::vector<char> name(MPI_MAX_PROCESSOR_NAME + 1, zero);
    // int resultlen = 0;
    // MPI_Get_processor_name(name.data(), &resultlen);
    gethostname(name.data(), MPI_MAX_PROCESSOR_NAME);
    std::string str(name.begin(), name.end());
    std::vector<char> allNames((MPI_MAX_PROCESSOR_NAME + 1) * size, zero);
    std::vector<size_t> aReceiveCount(size, MPI_MAX_PROCESSOR_NAME + 1);
    ccl::allgatherv((int8_t *)name.data(), name.size(),
                    (int8_t *)allNames.data(), aReceiveCount, comm)
        .wait();
    int localRank = 0;
    for (int i = 0; i < rank; i++) {
        auto nameBegin = allNames.begin() + i * (MPI_MAX_PROCESSOR_NAME + 1);
        std::string nbrName(nameBegin,
                            nameBegin + (MPI_MAX_PROCESSOR_NAME + 1));
        if (nbrName == str)
            localRank++;
    }
    return localRank;

    //    return 0;
}

sycl::device getAssignedGPU(ccl::communicator &comm, int size, int rankId,
                            jint *gpu_indices, int n_gpu) {
    auto local_rank = getLocalRank(comm, size, rankId);
    auto gpus = get_gpus();

    std::cout << "rank: " << rankId << " size: " << size
              << " local_rank: " << local_rank << " n_gpu: " << n_gpu
              << std::endl;

    auto gpu_selected = gpu_indices[local_rank % n_gpu];
    std::cout << "GPU selected for current rank: " << gpu_selected << std::endl;

    // In case gpu_selected index is larger than number of GPU SYCL devices
    auto rank_gpu = gpus[gpu_selected % gpus.size()];

    return rank_gpu;
}

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

template <typename T>
std::vector<oneapi::dal::table> split_table_by_rows(sycl::queue &queue,
                                                    const oneapi::dal::table &t,
                                                    std::int64_t split_count) {
    ONEDAL_ASSERT(split_count > 0);
    ONEDAL_ASSERT(split_count <= t.get_row_count());

    const std::int64_t row_count = t.get_row_count();
    const std::int64_t column_count = t.get_column_count();
    const std::int64_t block_size_regular = row_count / split_count;
    const std::int64_t block_size_tail = row_count % split_count;

    std::vector<oneapi::dal::table> result(split_count);

    std::int64_t row_offset = 0;
    for (std::int64_t i = 0; i < split_count; i++) {
        const std::int64_t tail =
            std::int64_t(i + 1 == split_count) * block_size_tail;
        const std::int64_t block_size = block_size_regular + tail;

        const auto row_range =
            oneapi::dal::range{row_offset, row_offset + block_size};
        const auto block = oneapi::dal::row_accessor<const T>{t}.pull(
            queue, row_range, sycl::usm::alloc::device);
        result[i] =
            oneapi::dal::homogen_table::wrap(block, block_size, column_count);
        row_offset += block_size;
    }

    return result;
}
