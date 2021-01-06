/// C++ extensions to support the metacells package.";

#if ASSERT_LEVEL > 0
#    undef NDEBUG
#    include <iostream>
#    include <mutex>
#elif ASSERT_LEVEL < 0 || ASSERT_LEVEL > 2
#    error Invalid ASSERT_LEVEL
#endif

#if ASSERT_LEVEL >= 1
#    define FastAssertCompare(X, OP, Y)                                                          \
        if (!(double(X) OP double(Y))) {                                                         \
            std::lock_guard<std::mutex> io_lock(io_mutex);                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << ": failed assert: " << #X << " -> " << X \
                      << " " << #OP << " " << Y << " <- " << #Y << "" << std::endl;              \
            assert(false);                                                                       \
        } else
#    define FastAssertCompareWhat(X, OP, Y, WHAT)                                                  \
        if (!(double(X) OP double(Y))) {                                                           \
            std::lock_guard<std::mutex> io_lock(io_mutex);                                         \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " << WHAT << ": failed assert: " << #X  \
                      << " -> " << X << " " << #OP << " " << Y << " <- " << #Y << "" << std::endl; \
            assert(false);                                                                         \
        } else
#else
#    define FastAssertCompare(...)
#    define FastAssertCompareWhat(...)
#endif

#if ASSERT_LEVEL >= 2
#    define SlowAssertCompare(...) FastAssertCompare(__VA_ARGS__)
#    define SlowAssertCompareWhat(...) FastAssertCompareWhat(__VA_ARGS__)
#else
#    define SlowAssertCompare(...)
#    define SlowAssertCompareWhat(...)
#endif

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <atomic>
#include <cmath>
#include <random>
#include <thread>

typedef float float32_t;
typedef double float64_t;

namespace metacells {

#if ASSERT_LEVEL > 0
static std::mutex io_mutex;
#endif

static_assert(sizeof(float32_t) == 4);
static_assert(sizeof(float64_t) == 8);

/// Release the GIL to allow for actual parallelism.
class WithoutGil {
private:
    PyThreadState* m_save;

public:
    WithoutGil() {
        Py_BEGIN_ALLOW_THREADS;
        m_save = _save;
    }
}

~WithoutGil(){ { PyThreadState* _save = m_save;
Py_END_ALLOW_THREADS;
}
}
;

static const double LOG2_SCALE = 1.0 / log(2.0);

static double
log2(const double x) {
    FastAssertCompare(x, >, 0);
    return log(x) * LOG2_SCALE;
}

/// An immutable contiguous slice of an array of type ``T``.
template<typename T>
class ConstArraySlice {
private:
    const T* m_data;     ///< Pointer to the first element.
    size_t m_size;       ///< Number of elements.
    const char* m_name;  ///< Name for error messages.

public:
    ConstArraySlice(const T* const data, const size_t size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    ConstArraySlice(const pybind11::array_t<T>& array, const char* const name)
      : ConstArraySlice(array.data(), array.size(), name) {
        FastAssertCompareWhat(array.ndim(), ==, 1, name);
        FastAssertCompareWhat(array.size(), >, 0, name);
        FastAssertCompareWhat(array.data(1) - array.data(0), ==, 1, name);
    }

    template<typename I>
    std::pair<ConstArraySlice, ConstArraySlice> split(const I size) const {
        return std::make_pair(slice(0, size), slice(size, m_size));
    }

    template<typename I, typename J>
    ConstArraySlice slice(const I start, const J stop) const {
        FastAssertCompareWhat(0, <=, start, m_name);
        FastAssertCompareWhat(start, <=, stop, m_name);
        FastAssertCompareWhat(stop, <=, m_size, m_name);
        return ConstArraySlice(m_data + start, stop - start, m_name);
    }

    size_t size() const { return m_size; }

    template<typename I>
    const T& operator[](const I index) const {
        SlowAssertCompareWhat(0, <=, index, m_name);
        SlowAssertCompareWhat(index, <, m_size, m_name);
        return m_data[index];
    }

    const T* begin() const { return m_data; }

    const T* end() const { return m_data + m_size; }
};

/// A mutable contiguous slice of an array of type ``T``.
template<typename T>
class ArraySlice {
private:
    T* m_data;           ///< Pointer to the first element.
    size_t m_size;       ///< Number of elements.
    const char* m_name;  ///< Name for error messages.

public:
    ArraySlice(T* const data, const size_t size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    ArraySlice(pybind11::array_t<T>& array, const char* const name)
      : ArraySlice(array.mutable_data(), array.size(), name) {
        FastAssertCompareWhat(array.ndim(), ==, 1, name);
        FastAssertCompareWhat(array.size(), >, 0, name);
        FastAssertCompareWhat(array.data(1) - array.data(0), ==, 1, name);
    }

    template<typename I>
    std::pair<ArraySlice, ArraySlice> split(const I size) {
        return std::make_pair(slice(0, size), slice(size, m_size));
    }

    template<typename I, typename J>
    ArraySlice slice(const I start, const J stop) {
        FastAssertCompareWhat(0, <=, start, m_name);
        FastAssertCompareWhat(start, <=, stop, m_name);
        FastAssertCompareWhat(stop, <=, m_size, m_name);
        return ArraySlice(m_data + start, stop - start, m_name);
    }

    size_t size() const { return m_size; }

    template<typename I>
    T& operator[](const I index) {
        SlowAssertCompareWhat(0, <=, index, m_name);
        SlowAssertCompareWhat(index, <, m_size, m_name);
        return m_data[index];
    }

    T* begin() { return m_data; }

    T* end() { return m_data + m_size; }

    operator ConstArraySlice<T>() const { return ConstArraySlice<T>(m_data, m_size, m_name); }
};

template<typename T>
static size_t
matrix_step(const pybind11::array_t<T>& array, const char* const name) {
    FastAssertCompareWhat(array.ndim(), ==, 2, name);
    FastAssertCompareWhat(array.shape(0), >, 0, name);
    FastAssertCompareWhat(array.shape(1), >, 0, name);
    return array.data(1, 0) - array.data(0, 0);
}

/// An immutable row-major slice of a matrix of type ``T``.
template<typename T>
class ConstMatrixSlice {
private:
    const T* m_data;         ///< Pointer to the first element.
    size_t m_rows_count;     ///< Number of rows.
    size_t m_columns_count;  ///< Number of columns.
    size_t m_rows_offset;    ///< Offset between start of rows.
    const char* m_name;      ///< Name for error messages.

public:
    ConstMatrixSlice(const T* const data,
                     const size_t rows_count,
                     const size_t columns_count,
                     const size_t rows_offset,
                     const char* const name)
      : m_data(data)
      , m_rows_count(rows_count)
      , m_columns_count(columns_count)
      , m_rows_offset(rows_offset)
      , m_name(name) {}

    ConstMatrixSlice(const pybind11::array_t<T>& array, const char* const name)
      : ConstMatrixSlice(array.data(),
                         array.shape(0),
                         array.shape(1),
                         matrix_step(array, name),
                         name) {
        FastAssertCompareWhat(array.ndim(), ==, 2, name);
        FastAssertCompareWhat(array.data(0, 1) - array.data(0, 0), ==, 1, name);
        FastAssertCompareWhat(m_columns_count, <=, m_rows_offset, name);
    }

    template<typename I>
    ConstArraySlice<T> get_row(I row_index) const {
        FastAssertCompareWhat(0, <=, row_index, m_name);
        FastAssertCompareWhat(row_index, <, m_rows_count, m_name);
        return ConstArraySlice<T>(m_data + row_index * m_rows_offset, m_columns_count, m_name);
    }

    size_t rows_count() const { return m_rows_count; }

    size_t columns_count() const { return m_columns_count; }
};

/// A mutable row-major slice of a matrix of type ``T``.
template<typename T>
class MatrixSlice {
private:
    T* m_data;               ///< Pointer to the first element.
    size_t m_rows_count;     ///< Number of rows.
    size_t m_columns_count;  ///< Number of columns.
    size_t m_rows_offset;    ///< Offset between start of rows.
    const char* m_name;      ///< Name for error messages.

public:
    MatrixSlice(T* const data,
                const size_t rows_count,
                const size_t columns_count,
                const size_t rows_offset,
                const char* const name)
      : m_data(data)
      , m_rows_count(rows_count)
      , m_columns_count(columns_count)
      , m_rows_offset(rows_offset)
      , m_name(name) {}

    MatrixSlice(pybind11::array_t<T>& array, const char* const name)
      : MatrixSlice(array.mutable_data(),
                    array.shape(0),
                    array.shape(1),
                    matrix_step(array, name),
                    name) {
        FastAssertCompareWhat(array.ndim(), ==, 2, name);
        FastAssertCompareWhat(array.data(0, 1) - array.data(0, 0), ==, 1, name);
        FastAssertCompareWhat(m_columns_count, <=, m_rows_offset, name);
    }

    template<typename I>
    ArraySlice<T> get_row(I row_index) const {
        FastAssertCompareWhat(0, <=, row_index, m_name);
        FastAssertCompareWhat(row_index, <, m_rows_count, m_name);
        return ArraySlice<T>(m_data + row_index * m_rows_offset, m_columns_count, m_name);
    }

    size_t rows_count() const { return m_rows_count; }

    size_t columns_count() const { return m_columns_count; }

    operator ConstMatrixSlice<T>() const {
        return ConstMatrixSlice<T>(m_data, m_rows_count, m_columns_count, m_rows_offset, m_name);
    }
};

/// An immutable CSR/CSC sparse matrix.
template<typename D, typename I, typename P>
class ConstCompressedMatrix {
private:
    ConstArraySlice<D> m_data;     ///< Non-zero data.
    ConstArraySlice<I> m_indices;  ///< Column indices.
    ConstArraySlice<P> m_indptr;   ///< First and last indices positions per band.
    size_t m_bands_count;          ///< Number of bands.
    size_t m_elements_count;       ///< Number of elements.
    const char* m_name;            ///< Name for error messages.

public:
    ConstCompressedMatrix(ConstArraySlice<D>&& data,
                          ConstArraySlice<I>&& indices,
                          ConstArraySlice<P>&& indptr,
                          const I elements_count,
                          const char* const name)
      : m_data(data)
      , m_indices(indices)
      , m_indptr(indptr)
      , m_bands_count(indptr.size() - 1)
      , m_elements_count(elements_count)
      , m_name(name) {
        FastAssertCompareWhat(m_indptr[m_bands_count], ==, indices.size(), m_name);
        FastAssertCompareWhat(m_indptr[m_bands_count], ==, data.size(), m_name);
    }

    ConstArraySlice<D> data() const { return m_data; }

    ConstArraySlice<I> indices() const { return m_indices; }

    ConstArraySlice<P> indptr() const { return m_indptr; }

    size_t bands_count() const { return m_bands_count; }

    size_t elements_count() const { return m_elements_count; }

    template<typename J>
    ConstArraySlice<I> get_band_indices(const J band_index) const {
        auto start_position = m_indptr[band_index];
        auto stop_position = m_indptr[band_index + 1];
        return m_indices.slice(start_position, stop_position);
    }

    template<typename J>
    ConstArraySlice<D> get_band_data(const J band_index) const {
        auto start_position = m_indptr[band_index];
        auto stop_position = m_indptr[band_index + 1];
        return m_data.slice(start_position, stop_position);
    }
};

/// A mutable CSR compressed matrix.
template<typename D, typename I, typename P>
class CompressedMatrix {
private:
    ArraySlice<D> m_data;     ///< Non-zero data.
    ArraySlice<I> m_indices;  ///< Column indices.
    ArraySlice<P> m_indptr;   ///< First and last indices positions per band.
    size_t m_bands_count;     ///< Number of bands.
    size_t m_elements_count;  ///< Number of elements.
    const char* m_name;       ///< Name for error messages.

public:
    CompressedMatrix(ArraySlice<D>&& data,
                     ArraySlice<I>&& indices,
                     ArraySlice<P>&& indptr,
                     const I elements_count,
                     const char* const name)
      : m_data(data)
      , m_indices(indices)
      , m_indptr(indptr)
      , m_bands_count(indptr.size() - 1)
      , m_elements_count(elements_count)
      , m_name(name) {
        FastAssertCompareWhat(m_indptr[m_bands_count], ==, indices.size(), m_name);
        FastAssertCompareWhat(m_indptr[m_bands_count], ==, data.size(), m_name);
    }

    size_t bands_count() const { return m_bands_count; }

    size_t elements_count() const { return m_elements_count; }

    ConstArraySlice<D> data() const { return m_data; }

    ConstArraySlice<D> indices() const { return m_indices; }

    ConstArraySlice<P> indptr() const { return m_indptr; }

    ArraySlice<D> data() { return m_data; }

    ArraySlice<I> indices() { return m_indices; }

    ArraySlice<P> indptr() { return m_indptr; }

    template<typename J>
    ArraySlice<I> get_band_indices(const J band_index) {
        auto start_position = m_indptr[band_index];
        auto stop_position = m_indptr[band_index + 1];
        return m_indices.slice(start_position, stop_position);
    }

    template<typename J>
    ArraySlice<D> get_band_data(const J band_index) {
        auto start_position = m_indptr[band_index];
        auto stop_position = m_indptr[band_index + 1];
        return m_data.slice(start_position, stop_position);
    }
};

/// How many parallel threads to use.
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

static void
parallel_loop(const size_t size,
              std::function<void(size_t)> parallel_body,
              std::function<void(size_t)> serial_body) {
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

static void
parallel_loop(const size_t size, std::function<void(size_t)> parallel_body) {
    parallel_loop(size, parallel_body, parallel_body);
}

template<typename T>
static T
ceil_power_of_two(const T size) {
    return 1 << T(ceil(log2(size)));
}

static size_t
downsample_tmp_size(const size_t size) {
    if (size <= 1) {
        return 0;
    }
    return 2 * ceil_power_of_two(size) - 1;
}

template<typename D>
static void
initialize_tree(ConstArraySlice<D> input, ArraySlice<size_t> tree) {
    FastAssertCompare(input.size(), >=, 2);

    size_t input_size = ceil_power_of_two(input.size());
    std::copy(input.begin(), input.end(), tree.begin());
    std::fill(tree.begin() + input.size(), tree.begin() + input_size, 0);

    while (input_size > 1) {
        auto slices = tree.split(input_size);
        auto input = slices.first;
        tree = slices.second;

        input_size /= 2;
        for (size_t index = 0; index < input_size; ++index) {
            const auto left = input[index * 2];
            const auto right = input[index * 2 + 1];
            tree[index] = left + right;

            SlowAssertCompare(left, >=, 0);
            SlowAssertCompare(right, >=, 0);
            SlowAssertCompare(tree[index], ==, size_t(left) + size_t(right));
        }
    }
    FastAssertCompare(tree.size(), ==, 1);
}

static size_t
random_sample(ArraySlice<size_t> tree, ssize_t random) {
    size_t size_of_level = 1;
    ssize_t base_of_level = tree.size() - 1;
    size_t index_in_level = 0;
    size_t index_in_tree = base_of_level + index_in_level;

    while (true) {
        SlowAssertCompare(index_in_tree, ==, base_of_level + index_in_level);
        FastAssertCompare(tree[index_in_tree], >, random);
        --tree[index_in_tree];
        size_of_level *= 2;
        base_of_level -= size_of_level;

        if (base_of_level < 0) {
            return index_in_level;
        }

        index_in_level *= 2;
        index_in_tree = base_of_level + index_in_level;
        ssize_t right_random = random - ssize_t(tree[index_in_tree]);

        SlowAssertCompare(tree[base_of_level + index_in_level]
                              + tree[base_of_level + index_in_level + 1],
                          ==,
                          tree[base_of_level + size_of_level + index_in_level / 2] + 1);

        if (right_random >= 0) {
            ++index_in_level;
            ++index_in_tree;
            SlowAssertCompare(index_in_level, <, size_of_level);
            random = right_random;
        }
    }
}

static thread_local std::vector<size_t> tmp_positions;
static thread_local std::vector<size_t> tmp_indices;
static thread_local std::vector<float64_t> tmp_data_one;
static thread_local std::vector<float64_t> tmp_data_two;
static thread_local std::vector<size_t> tmp_tree;

template<typename D, typename O>
static void
downsample_slice(ConstArraySlice<D> input,
                 ArraySlice<O> output,
                 const size_t samples,
                 const int random_seed) {
    FastAssertCompare(samples, >=, 0);
    FastAssertCompare(output.size(), ==, input.size());

    tmp_tree.resize(downsample_tmp_size(input.size()));
    ArraySlice<size_t> tree(&tmp_tree[0], tmp_tree.size(), "tree");

    if (input.size() == 0) {
        return;
    }

    if (input.size() == 1) {
        output[0] = double(samples) < double(input[0]) ? samples : input[0];
        return;
    }

    initialize_tree(input, tree);
    size_t& total = tree[tree.size() - 1];

    if (total <= samples) {
        if (static_cast<const void*>(output.begin()) != static_cast<const void*>(input.begin())) {
            std::copy(input.begin(), input.end(), output.begin());
        }
        return;
    }

    std::fill(output.begin(), output.end(), 0);

    std::minstd_rand random(random_seed);
    for (size_t index = 0; index < samples; ++index) {
        ++output[random_sample(tree, random() % total)];
    }
}

/// See the Python `metacell.utilities.computation.downsample_array` function.
template<typename D, typename O>
static void
downsample_array(const pybind11::array_t<D>& input_array,
                 pybind11::array_t<O>& output_array,
                 const size_t samples,
                 const int random_seed) {
    WithoutGil without_gil{};

    ConstArraySlice<D> input{ input_array, "input_array" };
    ArraySlice<O> output{ output_array, "output_array" };

    downsample_slice(input, output, samples, random_seed);
}

/// See the Python `metacell.utilities.computation.downsample_matrix` function.
template<typename D, typename O>
static void
downsample_matrix(const pybind11::array_t<D>& input_matrix,
                  pybind11::array_t<O>& output_array,
                  const size_t samples,
                  const int random_seed) {
    WithoutGil without_gil{};

    ConstMatrixSlice<D> input{ input_matrix, "input_matrix" };
    MatrixSlice<O> output{ output_array, "output_array" };

    parallel_loop(input.rows_count(), [&](size_t row_index) {
        int slice_seed = random_seed == 0 ? 0 : random_seed + row_index * 997;
        downsample_slice(input.get_row(row_index), output.get_row(row_index), samples, slice_seed);
    });
}

template<typename D, typename P, typename O>
static void
downsample_band(const size_t band_index,
                ConstArraySlice<D> input_data,
                ConstArraySlice<P> input_indptr,
                ArraySlice<O> output,
                const size_t samples,
                const int random_seed) {
    auto start_element_offset = input_indptr[band_index];
    auto stop_element_offset = input_indptr[band_index + 1];

    auto band_input = input_data.slice(start_element_offset, stop_element_offset);
    auto band_output = output.slice(start_element_offset, stop_element_offset);

    downsample_slice(band_input, band_output, samples, random_seed);
}

/// See the Python `metacell.utilities.computation.downsample_matrix` function.
template<typename D, typename P, typename O>
static void
downsample_compressed(const pybind11::array_t<D>& input_data_array,
                      const pybind11::array_t<P>& input_indptr_array,
                      pybind11::array_t<O>& output_array,
                      const size_t samples,
                      const int random_seed) {
    WithoutGil without_gil{};

    ConstArraySlice<D> input_data{ input_data_array, "input_data_array" };
    ConstArraySlice<P> input_indptr{ input_indptr_array, "input_indptr_array" };
    ArraySlice<O> output{ output_array, "output_array" };

    parallel_loop(input_indptr.size() - 1, [&](size_t band_index) {
        int band_seed = random_seed == 0 ? 0 : random_seed + band_index * 997;
        downsample_band(band_index, input_data, input_indptr, output, samples, band_seed);
    });
}

template<typename D, typename I, typename P>
static void
serial_collect_compressed_band(const size_t input_band_index,
                               ConstArraySlice<D> input_data,
                               ConstArraySlice<I> input_indices,
                               ConstArraySlice<P> input_indptr,
                               ArraySlice<D> output_data,
                               ArraySlice<I> output_indices,
                               ArraySlice<P> output_indptr) {
    size_t start_input_element_offset = input_indptr[input_band_index];
    size_t stop_input_element_offset = input_indptr[input_band_index + 1];

    FastAssertCompare(0, <=, start_input_element_offset);
    FastAssertCompare(start_input_element_offset, <=, stop_input_element_offset);
    FastAssertCompare(stop_input_element_offset, <=, input_data.size());

    size_t output_element_index = input_band_index;

    for (size_t input_element_offset = start_input_element_offset;
         input_element_offset < stop_input_element_offset;
         ++input_element_offset) {
        auto input_element_index = input_indices[input_element_offset];
        auto input_element_data = input_data[input_element_offset];

        auto output_band_index = input_element_index;
        auto output_element_data = input_element_data;

        auto output_element_offset = output_indptr[output_band_index]++;

        output_indices[output_element_offset] = output_element_index;
        output_data[output_element_offset] = output_element_data;
    }
}

template<typename D, typename I, typename P>
static void
parallel_collect_compressed_band(const size_t input_band_index,
                                 ConstArraySlice<D> input_data,
                                 ConstArraySlice<I> input_indices,
                                 ConstArraySlice<P> input_indptr,
                                 ArraySlice<D> output_data,
                                 ArraySlice<I> output_indices,
                                 ArraySlice<P> output_indptr) {
    size_t start_input_element_offset = input_indptr[input_band_index];
    size_t stop_input_element_offset = input_indptr[input_band_index + 1];

    FastAssertCompare(0, <=, start_input_element_offset);
    FastAssertCompare(start_input_element_offset, <=, stop_input_element_offset);
    FastAssertCompare(stop_input_element_offset, <=, input_data.size());

    size_t output_element_index = input_band_index;

    for (size_t input_element_offset = start_input_element_offset;
         input_element_offset < stop_input_element_offset;
         ++input_element_offset) {
        auto input_element_index = input_indices[input_element_offset];
        auto input_element_data = input_data[input_element_offset];

        auto output_band_index = input_element_index;
        auto output_element_data = input_element_data;

        auto atomic_output_element_offset =
            reinterpret_cast<std::atomic<P>*>(&output_indptr[output_band_index]);
        auto output_element_offset =
            atomic_output_element_offset->fetch_add(1, std::memory_order_relaxed);

        output_indices[output_element_offset] = output_element_index;
        output_data[output_element_offset] = output_element_data;
    }
}

/// See the Python `metacell.utilities.computation._relayout_compressed` function.
template<typename D, typename I, typename P>
static void
collect_compressed(const pybind11::array_t<D>& input_data_array,
                   const pybind11::array_t<I>& input_indices_array,
                   const pybind11::array_t<P>& input_indptr_array,
                   pybind11::array_t<D>& output_data_array,
                   pybind11::array_t<I>& output_indices_array,
                   pybind11::array_t<P>& output_indptr_array) {
    WithoutGil without_gil{};

    ConstArraySlice<D> input_data{ input_data_array, "input_data_array" };
    ConstArraySlice<I> input_indices{ input_indices_array, "input_indices_array" };
    ConstArraySlice<P> input_indptr{ input_indptr_array, "input_indptr_array" };

    FastAssertCompare(input_data.size(), ==, input_indptr[input_indptr.size() - 1]);
    FastAssertCompare(input_indices.size(), ==, input_data.size());

    ArraySlice<D> output_data{ output_data_array, "output_data_array" };
    ArraySlice<I> output_indices{ output_indices_array, "output_indices_array" };
    ArraySlice<P> output_indptr{ output_indptr_array, "output_indptr_array" };

    FastAssertCompare(output_data.size(), ==, input_data.size());
    FastAssertCompare(output_indices.size(), ==, input_indices.size());
    FastAssertCompare(output_indptr[output_indptr.size() - 1], <=, output_data.size());

    parallel_loop(input_indptr.size() - 1,
                  [&](size_t input_band_index) {
                      parallel_collect_compressed_band(input_band_index,
                                                       input_data,
                                                       input_indices,
                                                       input_indptr,
                                                       output_data,
                                                       output_indices,
                                                       output_indptr);
                  },
                  [&](size_t input_band_index) {
                      serial_collect_compressed_band(input_band_index,
                                                     input_data,
                                                     input_indices,
                                                     input_indptr,
                                                     output_data,
                                                     output_indices,
                                                     output_indptr);
                  });
}

template<typename D, typename I, typename P>
static void
sort_band(const size_t band_index, CompressedMatrix<D, I, P>& matrix) {
    if (matrix.indptr()[band_index] == matrix.indptr()[band_index + 1]) {
        return;
    }

    auto band_indices = matrix.get_band_indices(band_index);
    auto band_data = matrix.get_band_data(band_index);

    tmp_positions.resize(band_indices.size());
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);
    std::sort(tmp_positions.begin(),
              tmp_positions.end(),
              [&](const size_t left_position, const size_t right_position) {
                  auto left_index = band_indices[left_position];
                  auto right_index = band_indices[right_position];
                  return left_index < right_index;
              });

    tmp_indices.resize(tmp_positions.size());
    tmp_data_one.resize(tmp_positions.size());

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    const size_t tmp_size = tmp_positions.size();
    for (size_t location = 0; location < tmp_size; ++location) {
        size_t position = tmp_positions[location];
        tmp_indices[location] = band_indices[position];
        tmp_data_one[location] = band_data[position];
    }

    std::copy(tmp_indices.begin(), tmp_indices.end(), band_indices.begin());
    std::copy(tmp_data_one.begin(), tmp_data_one.end(), band_data.begin());
}

/// See the Python `metacell.utilities.computation._relayout_compressed` function.
template<typename D, typename I, typename P>
static void
sort_compressed_indices(pybind11::array_t<D>& data_array,
                        pybind11::array_t<I>& indices_array,
                        pybind11::array_t<P>& indptr_array,
                        const size_t elements_count) {
    WithoutGil without_gil{};

    CompressedMatrix<D, I, P> matrix(ArraySlice<D>(data_array, "data"),
                                     ArraySlice<I>(indices_array, "indices"),
                                     ArraySlice<P>(indptr_array, "indptr"),
                                     elements_count,
                                     "compressed");

    parallel_loop(matrix.bands_count(), [&](size_t band_index) { sort_band(band_index, matrix); });
}

static void
collect_outgoing_row(const size_t row_index,
                     const size_t degree,
                     ConstMatrixSlice<float32_t>& similarity_matrix,
                     ArraySlice<int32_t> output_indices,
                     ArraySlice<float32_t> output_ranks) {
    const size_t size = similarity_matrix.rows_count();
    const auto row_similarities = similarity_matrix.get_row(row_index);

    const size_t start_position = row_index * degree;
    const size_t stop_position = start_position + degree;

    auto row_indices = output_indices.slice(start_position, stop_position);
    auto row_ranks = output_ranks.slice(start_position, stop_position);

    if (degree < size - 1) {
        tmp_positions.resize(size - 1);
        std::iota(tmp_positions.begin(), tmp_positions.begin() + row_index, 0);
        std::iota(tmp_positions.begin() + row_index, tmp_positions.end(), row_index + 1);

        std::nth_element(tmp_positions.begin(),
                         tmp_positions.begin() + degree,
                         tmp_positions.end(),
                         [&](const size_t left_column_index, const size_t right_column_index) {
                             float32_t left_similarity = row_similarities[left_column_index];
                             float32_t right_similarity = row_similarities[right_column_index];
                             return left_similarity > right_similarity;
                         });

        std::copy(tmp_positions.begin(), tmp_positions.begin() + degree, row_indices.begin());
        std::sort(row_indices.begin(), row_indices.end());

    } else {
        std::iota(row_indices.begin(), row_indices.begin() + row_index, 0);
        std::iota(row_indices.begin() + row_index, row_indices.begin() + degree, row_index + 1);
    }

    tmp_positions.resize(degree);
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);
    std::sort(tmp_positions.begin(),
              tmp_positions.end(),
              [&](const size_t left_position, const size_t right_position) {
                  float32_t left_similarity = row_similarities[row_indices[left_position]];
                  float32_t right_similarity = row_similarities[row_indices[right_position]];
                  return left_similarity < right_similarity;
              });
#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t location = 0; location < degree; ++location) {
        size_t position = tmp_positions[location];
        row_ranks[position] = location + 1;
    }
}

/// See the Python `metacell.tools.knn_graph._rank_outgoing` function.
static void
collect_outgoing(const size_t degree,
                 const pybind11::array_t<float32_t>& input_similarity_matrix,
                 pybind11::array_t<int32_t>& output_indices_array,
                 pybind11::array_t<float32_t>& output_ranks_array) {
    WithoutGil without_gil{};

    ConstMatrixSlice<float32_t> similarity_matrix(input_similarity_matrix, "similarity_matrix");
    FastAssertCompareWhat(similarity_matrix.rows_count(),
                          ==,
                          similarity_matrix.columns_count(),
                          "similarity_matrix");
    const size_t size = similarity_matrix.rows_count();

    ArraySlice<int32_t> output_indices(output_indices_array, "output_indices");
    ArraySlice<float32_t> output_ranks(output_ranks_array, "output_ranks");

    FastAssertCompare(0, <, degree);
    FastAssertCompare(degree, <, size);

    FastAssertCompare(output_indices.size(), ==, degree * size);
    FastAssertCompare(output_ranks.size(), ==, degree * size);

    parallel_loop(size, [&](size_t row_index) {
        collect_outgoing_row(row_index, degree, similarity_matrix, output_indices, output_ranks);
    });
}

static void
prune_band(const size_t band_index,
           const size_t pruned_degree,
           ConstCompressedMatrix<float32_t, int32_t, int32_t>& input_pruned_ranks,
           ArraySlice<float32_t> output_pruned_ranks,
           ArraySlice<int32_t> output_pruned_indices,
           ConstArraySlice<int32_t> output_pruned_indptr) {
    const auto start_position = output_pruned_indptr[band_index];
    const auto stop_position = output_pruned_indptr[band_index + 1];

    auto output_indices = output_pruned_indices.slice(start_position, stop_position);
    auto output_ranks = output_pruned_ranks.slice(start_position, stop_position);

    const auto input_indices = input_pruned_ranks.get_band_indices(band_index);
    const auto input_ranks = input_pruned_ranks.get_band_data(band_index);
    FastAssertCompare(input_indices.size(), ==, input_ranks.size());
    FastAssertCompare(input_ranks.size(), ==, input_ranks.size());

    if (input_ranks.size() <= pruned_degree) {
        std::copy(input_indices.begin(), input_indices.end(), output_indices.begin());
        std::copy(input_ranks.begin(), input_ranks.end(), output_ranks.begin());
        return;
    }

    tmp_indices.resize(input_ranks.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);
    std::nth_element(tmp_indices.begin(),
                     tmp_indices.begin() + pruned_degree,
                     tmp_indices.end(),
                     [&](const size_t left_column_index, const size_t right_column_index) {
                         const auto left_similarity = input_ranks[left_column_index];
                         const auto right_similarity = input_ranks[right_column_index];
                         return left_similarity > right_similarity;
                     });

    tmp_indices.resize(pruned_degree);
    std::sort(tmp_indices.begin(), tmp_indices.end());

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t location = 0; location < pruned_degree; ++location) {
        size_t position = tmp_indices[location];
        output_indices[location] = input_indices[position];
        output_ranks[location] = input_ranks[position];
    }
}

/// See the Python `metacell.tools.knn_graph._prune_ranks` function.
static void
collect_pruned(const size_t pruned_degree,
               const pybind11::array_t<float32_t>& input_pruned_ranks_data,
               const pybind11::array_t<int32_t>& input_pruned_ranks_indices,
               const pybind11::array_t<int32_t>& input_pruned_ranks_indptr,
               pybind11::array_t<float32_t>& output_pruned_ranks_array,
               pybind11::array_t<int32_t>& output_pruned_indices_array,
               pybind11::array_t<int32_t>& output_pruned_indptr_array) {
    size_t size = input_pruned_ranks_indptr.size() - 1;
    ConstCompressedMatrix<float32_t, int32_t, int32_t> input_pruned_ranks(
        ConstArraySlice<float32_t>(input_pruned_ranks_data, "input_pruned_ranks_data"),
        ConstArraySlice<int32_t>(input_pruned_ranks_indices, "input_pruned_ranks_indices"),
        ConstArraySlice<int32_t>(input_pruned_ranks_indptr, "pruned_ranks_indptr"),
        size,
        "pruned_ranks");

    ArraySlice<float32_t> output_pruned_ranks(output_pruned_ranks_array, "output_pruned_ranks");
    ArraySlice<int32_t> output_pruned_indices(output_pruned_indices_array, "output_pruned_indices");
    ArraySlice<int32_t> output_pruned_indptr(output_pruned_indptr_array, "output_pruned_indptr");

    FastAssertCompare(output_pruned_ranks.size(), >=, size * pruned_degree);
    FastAssertCompare(output_pruned_indices.size(), >=, size * pruned_degree);
    FastAssertCompare(output_pruned_indptr.size(), ==, size + 1);

    size_t start_position = output_pruned_indptr[0] = 0;
    for (size_t band_index = 0; band_index < size; ++band_index) {
        FastAssertCompare(start_position, ==, output_pruned_indptr[band_index]);
        auto input_ranks = input_pruned_ranks.get_band_data(band_index);
        if (input_ranks.size() <= pruned_degree) {
            start_position += input_ranks.size();
        } else {
            start_position += pruned_degree;
        }
        output_pruned_indptr[band_index + 1] = start_position;
    }

    parallel_loop(size, [&](size_t band_index) {
        prune_band(band_index,
                   pruned_degree,
                   input_pruned_ranks,
                   output_pruned_ranks,
                   output_pruned_indices,
                   output_pruned_indptr);
    });
}

template<typename D, typename I, typename P>
static void
shuffle_band(const size_t band_index, CompressedMatrix<D, I, P>& matrix, const int random_seed) {
    std::minstd_rand random(random_seed);
    tmp_indices.resize(matrix.elements_count());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);
    std::random_shuffle(tmp_indices.begin(), tmp_indices.end(), [&](size_t n) {
        return random() % n;
    });
    auto band_indices = matrix.get_band_indices(band_index);
    tmp_indices.resize(band_indices.size());
    std::copy(tmp_indices.begin(), tmp_indices.end(), band_indices.begin());
    sort_band(band_index, matrix);
}

/// See the Python `metacell.utilities.computation.shuffle_matrix` function.
template<typename D, typename I, typename P>
static void
shuffle_compressed(pybind11::array_t<D>& data_array,
                   pybind11::array_t<I>& indices_array,
                   pybind11::array_t<P>& indptr_array,
                   const size_t elements_count,
                   const int random_seed) {
    CompressedMatrix<D, I, P> matrix(ArraySlice<D>(data_array, "data"),
                                     ArraySlice<I>(indices_array, "indices"),
                                     ArraySlice<P>(indptr_array, "indptr"),
                                     elements_count,
                                     "compressed");

    parallel_loop(matrix.bands_count(), [&](size_t band_index) {
        int band_seed = random_seed == 0 ? 0 : random_seed + band_index * 997;
        shuffle_band(band_index, matrix, band_seed);
    });
}

template<typename D>
static void
shuffle_row(const size_t row_index, MatrixSlice<D>& matrix, const int random_seed) {
    std::minstd_rand random(random_seed);
    auto row = matrix.get_row(row_index);
    std::random_shuffle(row.begin(), row.end(), [&](size_t n) { return random() % n; });
}

/// See the Python `metacell.utilities.computation.shuffle_matrix` function.
template<typename D>
static void
shuffle_matrix(pybind11::array_t<D>& matrix_array, const int random_seed) {
    MatrixSlice<D> matrix(matrix_array, "matrix");

    parallel_loop(matrix.rows_count(), [&](size_t row_index) {
        int row_seed = random_seed == 0 ? 0 : random_seed + row_index * 997;
        shuffle_row(row_index, matrix, row_seed);
    });
}

template<typename D>
static D
rank_row(size_t row_index, ConstMatrixSlice<D>& input, size_t rank) {
    const auto row_input = input.get_row(row_index);
    tmp_indices.resize(input.columns_count());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);
    std::nth_element(tmp_indices.begin(),
                     tmp_indices.begin() + rank,
                     tmp_indices.end(),
                     [&](const size_t left_column_index, const size_t right_column_index) {
                         const auto left_value = row_input[left_column_index];
                         const auto right_value = row_input[right_column_index];
                         return left_value < right_value;
                     });
    return row_input[tmp_indices[rank]];
}

/// See the Python `metacell.utilities.computation.rank_per` function.
template<typename D>
static void
rank_matrix(const pybind11::array_t<D>& input_matrix,
            pybind11::array_t<D>& output_array,
            const size_t rank) {
    ConstMatrixSlice<D> input(input_matrix, "input");
    ArraySlice<D> output(output_array, "array");

    const size_t rows_count = input.rows_count();
    FastAssertCompare(rows_count, ==, output_array.size());
    FastAssertCompare(rank, <, input.columns_count());

    parallel_loop(rows_count,
                  [&](size_t row_index) { output[row_index] = rank_row(row_index, input, rank); });
}

/// See the Python `metacell.tools.outlier_cells._collect_fold_factors` function.
template<typename D>
static void
fold_factor_dense(pybind11::array_t<D>& data_array,
                  const double min_gene_fold_factor,
                  const pybind11::array_t<D>& total_of_rows_array,
                  const pybind11::array_t<D>& fraction_of_columns_array) {
    MatrixSlice<D> data(data_array, "data");
    ConstArraySlice<D> total_of_rows(total_of_rows_array, "total_of_rows");
    ConstArraySlice<D> fraction_of_columns(fraction_of_columns_array, "fraction_of_columns");

    FastAssertCompare(total_of_rows.size(), ==, data.rows_count());
    FastAssertCompare(fraction_of_columns.size(), ==, data.columns_count());

    const size_t rows_count = data.rows_count();
    const size_t columns_count = data.columns_count();
    parallel_loop(rows_count, [&](size_t row_index) {
        const auto row_total = total_of_rows[row_index];
        auto row_data = data.get_row(row_index);
        for (size_t column_index = 0; column_index < columns_count; ++column_index) {
            const auto column_fraction = fraction_of_columns[column_index];
            const auto expected = row_total * column_fraction;
            auto& value = row_data[column_index];
            value = log((value + 1.0) / (expected + 1.0)) * LOG2_SCALE;
            if (value < min_gene_fold_factor) {
                value = 0.0;
            }
        }
    });
}

/// See the Python `metacell.tools.outlier_cells._collect_fold_factors` function.
template<typename D, typename I, typename P>
static void
fold_factor_compressed(pybind11::array_t<D>& data_array,
                       pybind11::array_t<I>& indices_array,
                       pybind11::array_t<P>& indptr_array,
                       const double min_gene_fold_factor,
                       const pybind11::array_t<D>& total_of_bands_array,
                       const pybind11::array_t<D>& fraction_of_elements_array) {
    ConstArraySlice<D> total_of_bands(total_of_bands_array, "total_of_bands");
    ConstArraySlice<D> fraction_of_elements(fraction_of_elements_array, "fraction_of_elements");

    const size_t bands_count = total_of_bands.size();
    const size_t elements_count = fraction_of_elements.size();

    CompressedMatrix<D, I, P> data(ArraySlice<D>(data_array, "data"),
                                   ArraySlice<I>(indices_array, "indices"),
                                   ArraySlice<P>(indptr_array, "indptr"),
                                   elements_count,
                                   "data");
    FastAssertCompare(data.bands_count(), ==, bands_count);
    FastAssertCompare(data.elements_count(), ==, elements_count);

    parallel_loop(bands_count, [&](size_t band_index) {
        const auto band_total = total_of_bands[band_index];
        auto band_indices = data.get_band_indices(band_index);
        auto band_data = data.get_band_data(band_index);

        const size_t band_elements_count = band_indices.size();
        for (size_t position = 0; position < band_elements_count; ++position) {
            const auto element_index = band_indices[position];
            const auto element_fraction = fraction_of_elements[element_index];
            const auto expected = band_total * element_fraction;
            auto& value = band_data[position];
            value = log((value + 1.0) / (expected + 1.0)) * LOG2_SCALE;
            if (value < min_gene_fold_factor) {
                value = 0.0;
            }
        }
    });
}

static void
collect_distinct_abs_folds(ArraySlice<int32_t> gene_indices,
                           ArraySlice<float32_t> gene_folds,
                           ConstArraySlice<float64_t> fold_in_cell) {
    tmp_indices.resize(fold_in_cell.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

    std::nth_element(tmp_indices.begin(),
                     tmp_indices.begin() + gene_indices.size(),
                     tmp_indices.end(),
                     [&](const size_t left_gene_index, const size_t right_gene_index) {
                         const auto left_value = fold_in_cell[left_gene_index];
                         const auto right_value = fold_in_cell[right_gene_index];
                         return abs(left_value) > abs(right_value);
                     });

    std::sort(tmp_indices.begin(),
              tmp_indices.begin() + gene_indices.size(),
              [&](const size_t left_gene_index, const size_t right_gene_index) {
                  const auto left_value = fold_in_cell[left_gene_index];
                  const auto right_value = fold_in_cell[right_gene_index];
                  return abs(left_value) > abs(right_value);
              });

    for (size_t position = 0; position < gene_indices.size(); ++position) {
        size_t gene_index = tmp_indices[position];
        gene_indices[position] = gene_index;
        gene_folds[position] = fold_in_cell[gene_index];
    }
}

static void
collect_distinct_high_folds(ArraySlice<int32_t> gene_indices,
                            ArraySlice<float32_t> gene_folds,
                            ConstArraySlice<float64_t> fold_in_cell) {
    tmp_indices.resize(fold_in_cell.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

    std::nth_element(tmp_indices.begin(),
                     tmp_indices.begin() + gene_indices.size(),
                     tmp_indices.end(),
                     [&](const size_t left_gene_index, const size_t right_gene_index) {
                         const auto left_value = fold_in_cell[left_gene_index];
                         const auto right_value = fold_in_cell[right_gene_index];
                         return left_value > right_value;
                     });

    std::sort(tmp_indices.begin(),
              tmp_indices.begin() + gene_indices.size(),
              [&](const size_t left_gene_index, const size_t right_gene_index) {
                  const auto left_value = fold_in_cell[left_gene_index];
                  const auto right_value = fold_in_cell[right_gene_index];
                  return left_value > right_value;
              });

    for (size_t position = 0; position < gene_indices.size(); ++position) {
        size_t gene_index = tmp_indices[position];
        gene_indices[position] = gene_index;
        gene_folds[position] = fold_in_cell[gene_index];
    }
}

static void
top_distinct(pybind11::array_t<int32_t>& gene_indices_array,
             pybind11::array_t<float32_t>& gene_folds_array,
             const pybind11::array_t<float64_t>& fold_in_cells_array,
             bool consider_low_folds) {
    MatrixSlice<float32_t> gene_folds(gene_folds_array, "gene_folds");
    MatrixSlice<int32_t> gene_indices(gene_indices_array, "gene_indices");
    ConstMatrixSlice<float64_t> fold_in_cells(fold_in_cells_array, "fold_in_cells");

    size_t cells_count = fold_in_cells.rows_count();
    size_t genes_count = fold_in_cells.columns_count();
    size_t distinct_count = gene_indices.columns_count();

    FastAssertCompare(distinct_count, <, genes_count);
    FastAssertCompare(gene_indices.rows_count(), ==, cells_count);
    FastAssertCompare(gene_folds.rows_count(), ==, cells_count);
    FastAssertCompare(gene_folds.columns_count(), ==, distinct_count);

    if (consider_low_folds) {
        parallel_loop(cells_count, [&](size_t cell_index) {
            collect_distinct_abs_folds(gene_indices.get_row(cell_index),
                                       gene_folds.get_row(cell_index),
                                       fold_in_cells.get_row(cell_index));
        });
    } else {
        parallel_loop(cells_count, [&](size_t cell_index) {
            collect_distinct_high_folds(gene_indices.get_row(cell_index),
                                        gene_folds.get_row(cell_index),
                                        fold_in_cells.get_row(cell_index));
        });
    }
}

static float64_t
auroc_data(std::vector<float64_t>& in_values,
           std::vector<float64_t>& out_values,
           const size_t zeros_count) {
    std::sort(in_values.rbegin(), in_values.rend());
    std::sort(out_values.rbegin(), out_values.rend());

    const size_t in_size = in_values.size();
    const size_t out_size = out_values.size();
    const size_t total_out_size = out_size + zeros_count;

    if (in_size == 0) {
        FastAssertCompare(total_out_size, >, 0);
        return 0.0;
    }

    if (out_size == 0) {
        FastAssertCompare(total_out_size, >, 0);
        return 1.0;
    }

    const float64_t in_scale = 1.0 / in_size;
    const float64_t out_scale = 1.0 / total_out_size;

    size_t in_count = 0;
    size_t out_count = 0;

    size_t in_index = 0;
    size_t out_index = 0;

    float64_t area = 0;

    do {
        float64_t value = std::max(in_values[in_index], out_values[out_index]);
        while (in_index < in_size && in_values[in_index] >= value)
            ++in_index;
        while (out_index < out_size && out_values[out_index] >= value)
            ++out_index;
        area += (out_index - out_count) * out_scale * (in_index + in_count) * in_scale / 2;
        in_count = in_index;
        out_count = out_index;
    } while (in_count < in_size && out_count < out_size);

    area += (total_out_size - out_count) * out_scale * (in_count + in_size) * in_scale / 2;

    return area;
}

template<typename D>
static float64_t
auroc_dense_vector(const ConstArraySlice<D>& values,
                   const ConstArraySlice<bool>& labels,
                   const ConstArraySlice<float32_t>& scales) {
    const size_t size = labels.size();
    FastAssertCompare(values.size(), ==, size);

    tmp_data_one.reserve(size);
    tmp_data_two.reserve(size);

    tmp_data_one.resize(0);
    tmp_data_two.resize(0);

    for (size_t index = 0; index < size; ++index) {
        const auto value = values[index] / scales[index];
        if (labels[index]) {
            tmp_data_one.push_back(value);
        } else {
            tmp_data_two.push_back(value);
        }
    }

    return auroc_data(tmp_data_one, tmp_data_two, 0);
}

template<typename D>
static void
auroc_dense_matrix(const pybind11::array_t<D>& values_array,
                   const pybind11::array_t<bool>& column_labels_array,
                   const pybind11::array_t<float32_t>& column_scales_array,
                   pybind11::array_t<float64_t>& aurocs_array) {
    ConstMatrixSlice<D> values(values_array, "values");
    ConstArraySlice<bool> column_labels(column_labels_array, "column_labels");
    ConstArraySlice<float32_t> column_scales(column_scales_array, "column_scales");
    ArraySlice<float64_t> row_aurocs(aurocs_array, "row_aurocs");

    const size_t columns_count = values.columns_count();
    const size_t rows_count = values.rows_count();

    FastAssertCompare(column_labels.size(), ==, columns_count);
    FastAssertCompare(row_aurocs.size(), ==, rows_count);

    parallel_loop(rows_count, [&](size_t row_index) {
        row_aurocs[row_index] =
            auroc_dense_vector(values.get_row(row_index), column_labels, column_scales);
    });
}

template<typename D, typename I>
static float64_t
auroc_compressed_vector(const ConstArraySlice<D>& values,
                        const ConstArraySlice<I>& indices,
                        const ConstArraySlice<bool>& labels,
                        const ConstArraySlice<float32_t>& scales) {
    const size_t size = labels.size();
    const size_t nnz_count = values.size();
    FastAssertCompare(nnz_count, <=, size);

    tmp_data_one.reserve(size);
    tmp_data_two.reserve(size);

    tmp_data_one.resize(0);
    tmp_data_two.resize(0);

    size_t prev_index = 0;
    for (size_t position = 0; position < nnz_count; ++position) {
        size_t index = size_t(indices[position]);
        auto value = values[position] / scales[index];

        FastAssertCompare(prev_index, <=, index);
        while (prev_index < index) {
            if (labels[prev_index]) {
                tmp_data_one.push_back(0.0);
            }
            ++prev_index;
        }

        if (labels[index]) {
            tmp_data_one.push_back(value);
        } else {
            tmp_data_two.push_back(value);
        }

        FastAssertCompare(prev_index, ==, index);
        ++prev_index;
    }

    FastAssertCompare(prev_index, <=, size);
    while (prev_index < size) {
        if (labels[prev_index]) {
            tmp_data_one.push_back(0.0);
        }
        ++prev_index;
    }

    FastAssertCompare(prev_index, ==, size);
    return auroc_data(tmp_data_one, tmp_data_two, size - nnz_count);
}

template<typename D, typename I, typename P>
static void
auroc_compressed_matrix(const pybind11::array_t<D>& values_data_array,
                        const pybind11::array_t<I>& values_indices_array,
                        const pybind11::array_t<P>& values_indptr_array,
                        size_t elements_count,
                        const pybind11::array_t<bool>& element_labels_array,
                        const pybind11::array_t<float32_t>& element_scales_array,
                        pybind11::array_t<float64_t>& band_aurocs_array) {
    ConstCompressedMatrix<D, I, P> values(ConstArraySlice<D>(values_data_array, "values_data"),
                                          ConstArraySlice<I>(values_indices_array,
                                                             "values_indices"),
                                          ConstArraySlice<P>(values_indptr_array, "values_indptr"),
                                          elements_count,
                                          "values");
    ConstArraySlice<bool> element_labels(element_labels_array, "element_labels");
    ConstArraySlice<float32_t> element_scales(element_scales_array, "element_scales");
    ArraySlice<float64_t> band_aurocs(band_aurocs_array, "band_aurocs");

    parallel_loop(values.bands_count(), [&](size_t band_index) {
        band_aurocs[band_index] = auroc_compressed_vector(values.get_band_data(band_index),
                                                          values.get_band_indices(band_index),
                                                          element_labels,
                                                          element_scales);
    });
}

template<typename D>
static float32_t
logistics_dense_vectors(ConstArraySlice<D> left,
                        ConstArraySlice<D> right,
                        const float64_t location,
                        const float64_t scale) {
    FastAssertCompare(right.size(), ==, left.size());

    const size_t size = left.size();

    float64_t result = 0;

    for (size_t index = 0; index < size; ++index) {
        float64_t diff = fabs(left[index] - right[index]);
        result += 1 / (1 + exp(scale * (location - diff)));
    }

    return float32_t(result / size);
}

template<typename D>
static void
logistics_dense_matrix(const pybind11::array_t<D>& values_array,
                       pybind11::array_t<float32_t>& logistics_array,
                       const float64_t location,
                       const float64_t scale) {
    ConstMatrixSlice<D> values(values_array, "values");
    MatrixSlice<float32_t> logistics(logistics_array, "logistics");

    const size_t rows_count = values.rows_count();

    FastAssertCompare(logistics.columns_count(), ==, rows_count);
    FastAssertCompare(logistics.rows_count(), ==, rows_count);

    const size_t iterations_count = (rows_count * (rows_count - 1)) / 2;

    for (size_t entry_index = 0; entry_index < rows_count; ++entry_index) {
        logistics.get_row(entry_index)[entry_index] = 0;
    }

    parallel_loop(iterations_count, [&](size_t iteration_index) {
        size_t some_index = iteration_index / (rows_count - 1);
        size_t other_index = iteration_index % (rows_count - 1);
        if (other_index < rows_count - 1 - some_index) {
            some_index = rows_count - 1 - some_index;
        } else {
            other_index = rows_count - 2 - other_index;
        }
        float logistic = logistics_dense_vectors(values.get_row(some_index),
                                                 values.get_row(other_index),
                                                 location,
                                                 scale);
        logistics.get_row(some_index)[other_index] = logistic;
        logistics.get_row(other_index)[some_index] = logistic;
    });
}

}  // namespace metacells

PYBIND11_MODULE(extensions, module) {
    module.doc() = "C++ extensions to support the metacells package.";

    module.def("set_threads_count",
               &metacells::set_threads_count,
               "Specify the number of parallel threads.");

#define REGISTER_D(D)                                                                        \
    module.def("shuffle_matrix_" #D, &metacells::shuffle_matrix<D>, "Shuffle matrix data."); \
    module.def("rank_matrix_" #D, &metacells::rank_matrix<D>, "Rank of matrix data.");       \
    module.def("fold_factor_dense_" #D,                                                      \
               &metacells::fold_factor_dense<D>,                                             \
               "Fold factors of dense data.");                                               \
    module.def("auroc_dense_matrix_" #D,                                                     \
               &metacells::auroc_dense_matrix<D>,                                            \
               "AUROC for dense matrix.");                                                   \
    module.def("logistics_dense_matrix_" #D,                                                 \
               &metacells::logistics_dense_matrix<D>,                                        \
               "Logistics distances for dense matrix.");

#define REGISTER_D_O(D, O)                          \
    module.def("downsample_array_" #D "_" #O,       \
               &metacells::downsample_array<D, O>,  \
               "Downsample array data.");           \
    module.def("downsample_matrix_" #D "_" #O,      \
               &metacells::downsample_matrix<D, O>, \
               "Downsample matrix data.");

#define REGISTER_D_P_O(D, P, O)                            \
    module.def("downsample_compressed_" #D "_" #P "_" #O,  \
               &metacells::downsample_compressed<D, P, O>, \
               "Downsample compressed data.");

#define REGISTER_D_I_P(D, I, P)                              \
    module.def("collect_compressed_" #D "_" #I "_" #P,       \
               &metacells::collect_compressed<D, I, P>,      \
               "Collect compressed data for relayout.");     \
    module.def("sort_compressed_indices_" #D "_" #I "_" #P,  \
               &metacells::sort_compressed_indices<D, I, P>, \
               "Sort indices in a compressed matrix.");      \
    module.def("shuffle_compressed_" #D "_" #I "_" #P,       \
               &metacells::shuffle_compressed<D, I, P>,      \
               "Shuffle compressed data.");                  \
    module.def("fold_factor_compressed_" #D "_" #I "_" #P,   \
               &metacells::fold_factor_compressed<D, I, P>,  \
               "Fold factors of compressed data.");          \
    module.def("auroc_compressed_matrix_" #D "_" #I "_" #P,  \
               &metacells::auroc_compressed_matrix<D, I, P>, \
               "AUROC for compressed matrix.");

    module.def("collect_outgoing",
               &metacells::collect_outgoing,
               "Collect the topmost outgoing edges.");

    module.def("collect_pruned", &metacells::collect_pruned, "Collect the topmost pruned edges.");
    module.def("top_distinct", &metacells::top_distinct, "Collect the topmost distinct genes.");

    REGISTER_D(float32_t)
    REGISTER_D(float64_t)
    REGISTER_D(int32_t)
    REGISTER_D(int64_t)
    REGISTER_D(uint32_t)
    REGISTER_D(uint64_t)

    REGISTER_D_O(float32_t, float32_t)
    REGISTER_D_O(float32_t, float64_t)
    REGISTER_D_O(float32_t, int32_t)
    REGISTER_D_O(float32_t, int64_t)
    REGISTER_D_O(float32_t, uint32_t)
    REGISTER_D_O(float32_t, uint64_t)
    REGISTER_D_O(float64_t, float32_t)
    REGISTER_D_O(float64_t, float64_t)
    REGISTER_D_O(float64_t, int32_t)
    REGISTER_D_O(float64_t, int64_t)
    REGISTER_D_O(float64_t, uint32_t)
    REGISTER_D_O(float64_t, uint64_t)
    REGISTER_D_O(int32_t, float32_t)
    REGISTER_D_O(int32_t, float64_t)
    REGISTER_D_O(int32_t, int32_t)
    REGISTER_D_O(int32_t, int64_t)
    REGISTER_D_O(int32_t, uint32_t)
    REGISTER_D_O(int32_t, uint64_t)
    REGISTER_D_O(int64_t, float32_t)
    REGISTER_D_O(int64_t, float64_t)
    REGISTER_D_O(int64_t, int32_t)
    REGISTER_D_O(int64_t, int64_t)
    REGISTER_D_O(int64_t, uint32_t)
    REGISTER_D_O(int64_t, uint64_t)
    REGISTER_D_O(uint32_t, float32_t)
    REGISTER_D_O(uint32_t, float64_t)
    REGISTER_D_O(uint32_t, int32_t)
    REGISTER_D_O(uint32_t, int64_t)
    REGISTER_D_O(uint32_t, uint32_t)
    REGISTER_D_O(uint32_t, uint64_t)
    REGISTER_D_O(uint64_t, float32_t)
    REGISTER_D_O(uint64_t, float64_t)
    REGISTER_D_O(uint64_t, int32_t)
    REGISTER_D_O(uint64_t, int64_t)
    REGISTER_D_O(uint64_t, uint32_t)
    REGISTER_D_O(uint64_t, uint64_t)

    REGISTER_D_P_O(float32_t, int32_t, float32_t)
    REGISTER_D_P_O(float32_t, int32_t, float64_t)
    REGISTER_D_P_O(float32_t, int32_t, int32_t)
    REGISTER_D_P_O(float32_t, int32_t, int64_t)
    REGISTER_D_P_O(float32_t, int32_t, uint32_t)
    REGISTER_D_P_O(float32_t, int32_t, uint64_t)
    REGISTER_D_P_O(float32_t, int64_t, float32_t)
    REGISTER_D_P_O(float32_t, int64_t, float64_t)
    REGISTER_D_P_O(float32_t, int64_t, int32_t)
    REGISTER_D_P_O(float32_t, int64_t, int64_t)
    REGISTER_D_P_O(float32_t, int64_t, uint32_t)
    REGISTER_D_P_O(float32_t, int64_t, uint64_t)
    REGISTER_D_P_O(float32_t, uint32_t, float32_t)
    REGISTER_D_P_O(float32_t, uint32_t, float64_t)
    REGISTER_D_P_O(float32_t, uint32_t, int32_t)
    REGISTER_D_P_O(float32_t, uint32_t, int64_t)
    REGISTER_D_P_O(float32_t, uint32_t, uint32_t)
    REGISTER_D_P_O(float32_t, uint32_t, uint64_t)
    REGISTER_D_P_O(float32_t, uint64_t, float32_t)
    REGISTER_D_P_O(float32_t, uint64_t, float64_t)
    REGISTER_D_P_O(float32_t, uint64_t, int64_t)
    REGISTER_D_P_O(float32_t, uint64_t, int64_t)
    REGISTER_D_P_O(float32_t, uint64_t, uint32_t)
    REGISTER_D_P_O(float32_t, uint64_t, uint64_t)
    REGISTER_D_P_O(float64_t, int32_t, float32_t)
    REGISTER_D_P_O(float64_t, int32_t, float64_t)
    REGISTER_D_P_O(float64_t, int32_t, int32_t)
    REGISTER_D_P_O(float64_t, int32_t, int64_t)
    REGISTER_D_P_O(float64_t, int32_t, uint32_t)
    REGISTER_D_P_O(float64_t, int32_t, uint64_t)
    REGISTER_D_P_O(float64_t, int64_t, float32_t)
    REGISTER_D_P_O(float64_t, int64_t, float64_t)
    REGISTER_D_P_O(float64_t, int64_t, int32_t)
    REGISTER_D_P_O(float64_t, int64_t, int64_t)
    REGISTER_D_P_O(float64_t, int64_t, uint32_t)
    REGISTER_D_P_O(float64_t, int64_t, uint64_t)
    REGISTER_D_P_O(float64_t, uint32_t, float32_t)
    REGISTER_D_P_O(float64_t, uint32_t, float64_t)
    REGISTER_D_P_O(float64_t, uint32_t, int32_t)
    REGISTER_D_P_O(float64_t, uint32_t, int64_t)
    REGISTER_D_P_O(float64_t, uint32_t, uint32_t)
    REGISTER_D_P_O(float64_t, uint32_t, uint64_t)
    REGISTER_D_P_O(float64_t, uint64_t, float32_t)
    REGISTER_D_P_O(float64_t, uint64_t, float64_t)
    REGISTER_D_P_O(float64_t, uint64_t, int64_t)
    REGISTER_D_P_O(float64_t, uint64_t, int64_t)
    REGISTER_D_P_O(float64_t, uint64_t, uint32_t)
    REGISTER_D_P_O(float64_t, uint64_t, uint64_t)
    REGISTER_D_P_O(int32_t, int32_t, float32_t)
    REGISTER_D_P_O(int32_t, int32_t, float64_t)
    REGISTER_D_P_O(int32_t, int32_t, int32_t)
    REGISTER_D_P_O(int32_t, int32_t, int64_t)
    REGISTER_D_P_O(int32_t, int32_t, uint32_t)
    REGISTER_D_P_O(int32_t, int32_t, uint64_t)
    REGISTER_D_P_O(int32_t, int64_t, float32_t)
    REGISTER_D_P_O(int32_t, int64_t, float64_t)
    REGISTER_D_P_O(int32_t, int64_t, int32_t)
    REGISTER_D_P_O(int32_t, int64_t, int64_t)
    REGISTER_D_P_O(int32_t, int64_t, uint32_t)
    REGISTER_D_P_O(int32_t, int64_t, uint64_t)
    REGISTER_D_P_O(int32_t, uint32_t, float32_t)
    REGISTER_D_P_O(int32_t, uint32_t, float64_t)
    REGISTER_D_P_O(int32_t, uint32_t, int32_t)
    REGISTER_D_P_O(int32_t, uint32_t, int64_t)
    REGISTER_D_P_O(int32_t, uint32_t, uint32_t)
    REGISTER_D_P_O(int32_t, uint32_t, uint64_t)
    REGISTER_D_P_O(int32_t, uint64_t, float32_t)
    REGISTER_D_P_O(int32_t, uint64_t, float64_t)
    REGISTER_D_P_O(int32_t, uint64_t, int64_t)
    REGISTER_D_P_O(int32_t, uint64_t, int64_t)
    REGISTER_D_P_O(int32_t, uint64_t, uint32_t)
    REGISTER_D_P_O(int32_t, uint64_t, uint64_t)
    REGISTER_D_P_O(int64_t, int32_t, float32_t)
    REGISTER_D_P_O(int64_t, int32_t, float64_t)
    REGISTER_D_P_O(int64_t, int32_t, int32_t)
    REGISTER_D_P_O(int64_t, int32_t, int64_t)
    REGISTER_D_P_O(int64_t, int32_t, uint32_t)
    REGISTER_D_P_O(int64_t, int32_t, uint64_t)
    REGISTER_D_P_O(int64_t, int64_t, float32_t)
    REGISTER_D_P_O(int64_t, int64_t, float64_t)
    REGISTER_D_P_O(int64_t, int64_t, int32_t)
    REGISTER_D_P_O(int64_t, int64_t, int64_t)
    REGISTER_D_P_O(int64_t, int64_t, uint32_t)
    REGISTER_D_P_O(int64_t, int64_t, uint64_t)
    REGISTER_D_P_O(int64_t, uint32_t, float32_t)
    REGISTER_D_P_O(int64_t, uint32_t, float64_t)
    REGISTER_D_P_O(int64_t, uint32_t, int32_t)
    REGISTER_D_P_O(int64_t, uint32_t, int64_t)
    REGISTER_D_P_O(int64_t, uint32_t, uint32_t)
    REGISTER_D_P_O(int64_t, uint32_t, uint64_t)
    REGISTER_D_P_O(int64_t, uint64_t, float32_t)
    REGISTER_D_P_O(int64_t, uint64_t, float64_t)
    REGISTER_D_P_O(int64_t, uint64_t, int64_t)
    REGISTER_D_P_O(int64_t, uint64_t, int64_t)
    REGISTER_D_P_O(int64_t, uint64_t, uint32_t)
    REGISTER_D_P_O(int64_t, uint64_t, uint64_t)
    REGISTER_D_P_O(uint32_t, int32_t, float32_t)
    REGISTER_D_P_O(uint32_t, int32_t, float64_t)
    REGISTER_D_P_O(uint32_t, int32_t, int32_t)
    REGISTER_D_P_O(uint32_t, int32_t, int64_t)
    REGISTER_D_P_O(uint32_t, int32_t, uint32_t)
    REGISTER_D_P_O(uint32_t, int32_t, uint64_t)
    REGISTER_D_P_O(uint32_t, int64_t, float32_t)
    REGISTER_D_P_O(uint32_t, int64_t, float64_t)
    REGISTER_D_P_O(uint32_t, int64_t, int32_t)
    REGISTER_D_P_O(uint32_t, int64_t, int64_t)
    REGISTER_D_P_O(uint32_t, int64_t, uint32_t)
    REGISTER_D_P_O(uint32_t, int64_t, uint64_t)
    REGISTER_D_P_O(uint32_t, uint32_t, float32_t)
    REGISTER_D_P_O(uint32_t, uint32_t, float64_t)
    REGISTER_D_P_O(uint32_t, uint32_t, int32_t)
    REGISTER_D_P_O(uint32_t, uint32_t, int64_t)
    REGISTER_D_P_O(uint32_t, uint32_t, uint32_t)
    REGISTER_D_P_O(uint32_t, uint32_t, uint64_t)
    REGISTER_D_P_O(uint32_t, uint64_t, float32_t)
    REGISTER_D_P_O(uint32_t, uint64_t, float64_t)
    REGISTER_D_P_O(uint32_t, uint64_t, int64_t)
    REGISTER_D_P_O(uint32_t, uint64_t, int64_t)
    REGISTER_D_P_O(uint32_t, uint64_t, uint32_t)
    REGISTER_D_P_O(uint32_t, uint64_t, uint64_t)
    REGISTER_D_P_O(uint64_t, int32_t, float32_t)
    REGISTER_D_P_O(uint64_t, int32_t, float64_t)
    REGISTER_D_P_O(uint64_t, int32_t, int32_t)
    REGISTER_D_P_O(uint64_t, int32_t, int64_t)
    REGISTER_D_P_O(uint64_t, int32_t, uint32_t)
    REGISTER_D_P_O(uint64_t, int32_t, uint64_t)
    REGISTER_D_P_O(uint64_t, int64_t, float32_t)
    REGISTER_D_P_O(uint64_t, int64_t, float64_t)
    REGISTER_D_P_O(uint64_t, int64_t, int32_t)
    REGISTER_D_P_O(uint64_t, int64_t, int64_t)
    REGISTER_D_P_O(uint64_t, int64_t, uint32_t)
    REGISTER_D_P_O(uint64_t, int64_t, uint64_t)
    REGISTER_D_P_O(uint64_t, uint32_t, float32_t)
    REGISTER_D_P_O(uint64_t, uint32_t, float64_t)
    REGISTER_D_P_O(uint64_t, uint32_t, int32_t)
    REGISTER_D_P_O(uint64_t, uint32_t, int64_t)
    REGISTER_D_P_O(uint64_t, uint32_t, uint32_t)
    REGISTER_D_P_O(uint64_t, uint32_t, uint64_t)
    REGISTER_D_P_O(uint64_t, uint64_t, float32_t)
    REGISTER_D_P_O(uint64_t, uint64_t, float64_t)
    REGISTER_D_P_O(uint64_t, uint64_t, int64_t)
    REGISTER_D_P_O(uint64_t, uint64_t, int64_t)
    REGISTER_D_P_O(uint64_t, uint64_t, uint32_t)
    REGISTER_D_P_O(uint64_t, uint64_t, uint64_t)

    REGISTER_D_I_P(float32_t, int32_t, int32_t)
    REGISTER_D_I_P(float32_t, int32_t, int64_t)
    REGISTER_D_I_P(float32_t, int32_t, uint32_t)
    REGISTER_D_I_P(float32_t, int32_t, uint64_t)
    REGISTER_D_I_P(float32_t, int64_t, int32_t)
    REGISTER_D_I_P(float32_t, int64_t, int64_t)
    REGISTER_D_I_P(float32_t, int64_t, uint32_t)
    REGISTER_D_I_P(float32_t, int64_t, uint64_t)
    REGISTER_D_I_P(float32_t, uint32_t, int32_t)
    REGISTER_D_I_P(float32_t, uint32_t, int64_t)
    REGISTER_D_I_P(float32_t, uint32_t, uint32_t)
    REGISTER_D_I_P(float32_t, uint32_t, uint64_t)
    REGISTER_D_I_P(float32_t, uint64_t, int32_t)
    REGISTER_D_I_P(float32_t, uint64_t, int64_t)
    REGISTER_D_I_P(float32_t, uint64_t, uint32_t)
    REGISTER_D_I_P(float32_t, uint64_t, uint64_t)
    REGISTER_D_I_P(float64_t, int32_t, int32_t)
    REGISTER_D_I_P(float64_t, int32_t, int64_t)
    REGISTER_D_I_P(float64_t, int32_t, uint32_t)
    REGISTER_D_I_P(float64_t, int32_t, uint64_t)
    REGISTER_D_I_P(float64_t, int64_t, int32_t)
    REGISTER_D_I_P(float64_t, int64_t, int64_t)
    REGISTER_D_I_P(float64_t, int64_t, uint32_t)
    REGISTER_D_I_P(float64_t, int64_t, uint64_t)
    REGISTER_D_I_P(float64_t, uint32_t, int32_t)
    REGISTER_D_I_P(float64_t, uint32_t, int64_t)
    REGISTER_D_I_P(float64_t, uint32_t, uint32_t)
    REGISTER_D_I_P(float64_t, uint32_t, uint64_t)
    REGISTER_D_I_P(float64_t, uint64_t, int32_t)
    REGISTER_D_I_P(float64_t, uint64_t, int64_t)
    REGISTER_D_I_P(float64_t, uint64_t, uint32_t)
    REGISTER_D_I_P(float64_t, uint64_t, uint64_t)
    REGISTER_D_I_P(int32_t, int32_t, int32_t)
    REGISTER_D_I_P(int32_t, int32_t, int64_t)
    REGISTER_D_I_P(int32_t, int32_t, uint32_t)
    REGISTER_D_I_P(int32_t, int32_t, uint64_t)
    REGISTER_D_I_P(int32_t, int64_t, int32_t)
    REGISTER_D_I_P(int32_t, int64_t, int64_t)
    REGISTER_D_I_P(int32_t, int64_t, uint32_t)
    REGISTER_D_I_P(int32_t, int64_t, uint64_t)
    REGISTER_D_I_P(int32_t, uint32_t, int32_t)
    REGISTER_D_I_P(int32_t, uint32_t, int64_t)
    REGISTER_D_I_P(int32_t, uint32_t, uint32_t)
    REGISTER_D_I_P(int32_t, uint32_t, uint64_t)
    REGISTER_D_I_P(int32_t, uint64_t, int32_t)
    REGISTER_D_I_P(int32_t, uint64_t, int64_t)
    REGISTER_D_I_P(int32_t, uint64_t, uint32_t)
    REGISTER_D_I_P(int32_t, uint64_t, uint64_t)
    REGISTER_D_I_P(int64_t, int32_t, int32_t)
    REGISTER_D_I_P(int64_t, int32_t, int64_t)
    REGISTER_D_I_P(int64_t, int32_t, uint32_t)
    REGISTER_D_I_P(int64_t, int32_t, uint64_t)
    REGISTER_D_I_P(int64_t, int64_t, int32_t)
    REGISTER_D_I_P(int64_t, int64_t, int64_t)
    REGISTER_D_I_P(int64_t, int64_t, uint32_t)
    REGISTER_D_I_P(int64_t, int64_t, uint64_t)
    REGISTER_D_I_P(int64_t, uint32_t, int32_t)
    REGISTER_D_I_P(int64_t, uint32_t, int64_t)
    REGISTER_D_I_P(int64_t, uint32_t, uint32_t)
    REGISTER_D_I_P(int64_t, uint32_t, uint64_t)
    REGISTER_D_I_P(int64_t, uint64_t, int32_t)
    REGISTER_D_I_P(int64_t, uint64_t, int64_t)
    REGISTER_D_I_P(int64_t, uint64_t, uint32_t)
    REGISTER_D_I_P(int64_t, uint64_t, uint64_t)
    REGISTER_D_I_P(uint32_t, int32_t, int32_t)
    REGISTER_D_I_P(uint32_t, int32_t, int64_t)
    REGISTER_D_I_P(uint32_t, int32_t, uint32_t)
    REGISTER_D_I_P(uint32_t, int32_t, uint64_t)
    REGISTER_D_I_P(uint32_t, int64_t, int32_t)
    REGISTER_D_I_P(uint32_t, int64_t, int64_t)
    REGISTER_D_I_P(uint32_t, int64_t, uint32_t)
    REGISTER_D_I_P(uint32_t, int64_t, uint64_t)
    REGISTER_D_I_P(uint32_t, uint32_t, int32_t)
    REGISTER_D_I_P(uint32_t, uint32_t, int64_t)
    REGISTER_D_I_P(uint32_t, uint32_t, uint32_t)
    REGISTER_D_I_P(uint32_t, uint32_t, uint64_t)
    REGISTER_D_I_P(uint32_t, uint64_t, int32_t)
    REGISTER_D_I_P(uint32_t, uint64_t, int64_t)
    REGISTER_D_I_P(uint32_t, uint64_t, uint32_t)
    REGISTER_D_I_P(uint32_t, uint64_t, uint64_t)
    REGISTER_D_I_P(uint64_t, int32_t, int32_t)
    REGISTER_D_I_P(uint64_t, int32_t, int64_t)
    REGISTER_D_I_P(uint64_t, int32_t, uint32_t)
    REGISTER_D_I_P(uint64_t, int32_t, uint64_t)
    REGISTER_D_I_P(uint64_t, int64_t, int32_t)
    REGISTER_D_I_P(uint64_t, int64_t, int64_t)
    REGISTER_D_I_P(uint64_t, int64_t, uint32_t)
    REGISTER_D_I_P(uint64_t, int64_t, uint64_t)
    REGISTER_D_I_P(uint64_t, uint32_t, int32_t)
    REGISTER_D_I_P(uint64_t, uint32_t, int64_t)
    REGISTER_D_I_P(uint64_t, uint32_t, uint32_t)
    REGISTER_D_I_P(uint64_t, uint32_t, uint64_t)
    REGISTER_D_I_P(uint64_t, uint64_t, int32_t)
    REGISTER_D_I_P(uint64_t, uint64_t, int64_t)
    REGISTER_D_I_P(uint64_t, uint64_t, uint32_t)
    REGISTER_D_I_P(uint64_t, uint64_t, uint64_t)
}
