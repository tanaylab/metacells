#include "metacells/extensions.h"

namespace metacells {

std::mutex writer_mutex;

thread_local AtomicWriter writer;

#if ASSERT_LEVEL > 0
std::mutex io_mutex;
#endif

static size_t threads_count = 1;

static void
set_threads_count(size_t count) {
    threads_count = count;
}

static std::atomic<size_t> next_loop_index;
static size_t loop_size;

const void
worker(std::function<void(size_t)> parallel_body) {
    for (size_t index = next_loop_index++; index < loop_size; index = next_loop_index++) {
        parallel_body(index);
    }
}

void
parallel_loop(const size_t size, std::function<void(size_t)> parallel_body, std::function<void(size_t)> serial_body) {
    size_t used_threads_count = std::min(threads_count, size);

    if (used_threads_count < 2) {
        for (size_t index = 0; index < size; ++index) {
            serial_body(index);
        }
        return;
    }

    next_loop_index = 0;
    loop_size = size;

    std::vector<std::thread> used_threads;
    used_threads.reserve(used_threads_count);

    while (next_loop_index < loop_size && used_threads.size() < used_threads_count) {
        used_threads.emplace_back(worker, parallel_body);
    }

    for (auto& thread : used_threads) {
        thread.join();
    }
}

thread_local bool g_size_t_used[size_t_count];
thread_local std::vector<size_t> g_size_t_vectors[size_t_count];

thread_local bool g_float64_t_used[float64_t_count];
thread_local std::vector<float64_t> g_float64_t_vectors[float64_t_count];

}  // namespace metacells

PYBIND11_MODULE(extensions, module) {
    module.doc() = "C++ extensions to support the metacells package.";

    module.def("set_threads_count", &metacells::set_threads_count, "Specify the number of parallel threads.");

    metacells::register_auroc(module);
    metacells::register_choose_seeds(module);
    metacells::register_correlate(module);
    metacells::register_cover(module);
    metacells::register_downsample(module);
    metacells::register_folds(module);
    metacells::register_logistics(module);
    metacells::register_partitions(module);
    metacells::register_prune_per(module);
    metacells::register_rank(module);
    metacells::register_relayout(module);
    metacells::register_shuffle(module);
    metacells::register_top_per(module);
}
