/// C++ extensions to support the metacells package.";

#include <iostream>

#if ASSERT_LEVEL > 0
#    undef NDEBUG
#    include <mutex>
#    include <sstream>
#    include <thread>

static std::mutex writer_mutex;

class AtomicWriter {
    std::ostringstream m_st;
    std::ostream& m_stream;

public:
    AtomicWriter(std::ostream& s = std::cerr) : m_stream(s) {
        m_st << std::this_thread::get_id() << ' ';
    }

    template<typename T>
    AtomicWriter& operator<<(T const& t) {
        m_st << t;
        return *this;
    }

    AtomicWriter& operator<<(std::ostream& (*f)(std::ostream&)) {
        m_st << f;
        {
            std::lock_guard<std::mutex> lock(writer_mutex);
            m_stream << m_st.str() << std::flush;
        }
        m_st.str("");
        m_st << std::this_thread::get_id() << ' ';
        return *this;
    }
};

static thread_local AtomicWriter writer;

#elif ASSERT_LEVEL < 0 || ASSERT_LEVEL > 2
#    error Invalid ASSERT_LEVEL
#endif

#define LOCATED_LOG(COND) \
    if (!(COND))          \
        ;                 \
    else                  \
        writer << __FILE__ << ':' << __LINE__ << ':' << __FUNCTION__ << ":"

#if ASSERT_LEVEL >= 1
#    define FastAssertCompare(X, OP, Y)                                                            \
        if (!(double(X) OP double(Y))) {                                                           \
            std::lock_guard<std::mutex> io_lock(io_mutex);                                         \
            std::cerr << __FILE__ << ":" << __LINE__ << ": failed assert: " << #X << " -> " << (X) \
                      << " " << #OP << " " << (Y) << " <- " << #Y << "" << std::endl;              \
            assert(false);                                                                         \
        } else
#    define FastAssertCompareWhat(X, OP, Y, WHAT)                                                 \
        if (!(double(X) OP double(Y))) {                                                          \
            std::lock_guard<std::mutex> io_lock(io_mutex);                                        \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " << WHAT << ": failed assert: " << #X \
                      << " -> " << (X) << " " << #OP << " " << (Y) << " <- " << #Y << ""          \
                      << std::endl;                                                               \
            assert(false);                                                                        \
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
typedef unsigned char uint8_t;
typedef unsigned int uint_t;

namespace metacells {

#if ASSERT_LEVEL > 0
static std::mutex io_mutex;
#endif

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

static const float64_t EPSILON = 1e-6;
static const float64_t LOG2_SCALE = 1.0 / log(2.0);
static const double NaN = std::numeric_limits<double>::quiet_NaN();

static float64_t
log2(const float64_t x) {
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
    ConstArraySlice(const std::vector<T>& vector, const char* const name)
      : m_data(&vector[0]), m_size(vector.size()), m_name(name) {}

    ConstArraySlice(const T* const data, const size_t size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    ConstArraySlice(const pybind11::array_t<T>& array, const char* const name)
      : ConstArraySlice(array.data(), array.size(), name) {
        FastAssertCompareWhat(array.ndim(), ==, 1, name);
        FastAssertCompareWhat(array.size(), >, 0, name);
        FastAssertCompareWhat(array.data(1) - array.data(0), ==, 1, name);
    }

    std::pair<ConstArraySlice, ConstArraySlice> split(const size_t size) const {
        return std::make_pair(slice(0, size), slice(size, m_size));
    }

    ConstArraySlice slice(const size_t start, const size_t stop) const {
        FastAssertCompareWhat(0, <=, start, m_name);
        FastAssertCompareWhat(start, <=, stop, m_name);
        FastAssertCompareWhat(stop, <=, m_size, m_name);
        return ConstArraySlice(m_data + start, stop - start, m_name);
    }

    size_t size() const { return m_size; }

    const T& operator[](const size_t index) const {
        SlowAssertCompareWhat(0, <=, index, m_name);
        SlowAssertCompareWhat(index, <, m_size, m_name);
        return m_data[index];
    }

    const T* begin() const { return m_data; }

    const T* end() const { return m_data + m_size; }

    bool sorted_contains(T item) const { return std::binary_search(begin(), end(), item); }
};

/// A mutable contiguous slice of an array of type ``T``.
template<typename T>
class ArraySlice {
private:
    T* m_data;           ///< Pointer to the first element.
    size_t m_size;       ///< Number of elements.
    const char* m_name;  ///< Name for error messages.

public:
    ArraySlice(std::vector<T>& vector, const char* const name)
      : m_data(&vector[0]), m_size(vector.size()), m_name(name) {}

    ArraySlice(T* const data, const size_t size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    ArraySlice(pybind11::array_t<T>& array, const char* const name)
      : ArraySlice(array.mutable_data(), array.size(), name) {
        FastAssertCompareWhat(array.ndim(), ==, 1, name);
        FastAssertCompareWhat(array.size(), >, 0, name);
        FastAssertCompareWhat(array.data(1) - array.data(0), ==, 1, name);
    }

    std::pair<ArraySlice, ArraySlice> split(const size_t size) {
        return std::make_pair(slice(0, size), slice(size, m_size));
    }

    ArraySlice slice(const size_t start, const size_t stop) {
        FastAssertCompareWhat(0, <=, start, m_name);
        FastAssertCompareWhat(start, <=, stop, m_name);
        FastAssertCompareWhat(stop, <=, m_size, m_name);
        return ArraySlice(m_data + start, stop - start, m_name);
    }

    size_t size() const { return m_size; }

    T& operator[](const size_t index) {
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

    ConstArraySlice<T> get_row(size_t row_index) const {
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

    ArraySlice<T> get_row(size_t row_index) const {
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
                          const size_t elements_count,
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

    ConstArraySlice<I> get_band_indices(const size_t band_index) const {
        auto start_position = m_indptr[band_index];
        auto stop_position = m_indptr[band_index + 1];
        return m_indices.slice(start_position, stop_position);
    }

    ConstArraySlice<D> get_band_data(const size_t band_index) const {
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
                     const size_t elements_count,
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

    ArraySlice<I> get_band_indices(const size_t band_index) {
        auto start_position = m_indptr[band_index];
        auto stop_position = m_indptr[band_index + 1];
        return m_indices.slice(start_position, stop_position);
    }

    ArraySlice<D> get_band_data(const size_t band_index) {
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

/*
static void
serial_loop(const size_t size, std::function<void(size_t)> serial_body) {
    for (size_t index = 0; index < size; ++index) {
        serial_body(index);
    }
}
*/

static size_t
ceil_power_of_two(const size_t size) {
    return size_t(1) << size_t(ceil(log2(float64_t(size))));
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

static const int size_t_count = 8;
static thread_local bool g_size_t_used[size_t_count];
static thread_local std::vector<size_t> g_size_t_vectors[size_t_count];

class TmpVectorSizeT {
private:
    int m_index;

public:
    TmpVectorSizeT() : m_index(-1) {
        for (int index = 0; index < size_t_count; ++index) {
            if (!g_size_t_used[index]) {
                g_size_t_used[index] = true;
                m_index = index;
                return;
            }
        }
        assert(false);
    }

    ~TmpVectorSizeT() {
        g_size_t_vectors[m_index].clear();
        g_size_t_used[m_index] = false;
    }

    std::vector<size_t>& vector(size_t size = 0) {
        g_size_t_vectors[m_index].resize(size);
        return g_size_t_vectors[m_index];
    }

    ArraySlice<size_t> array_slice(const char* const name, size_t size = 0) {
        return ArraySlice<size_t>(vector(size), name);
    }
};

static const int float64_t_count = 8;
static thread_local bool g_float64_t_used[float64_t_count];
static thread_local std::vector<float64_t> g_float64_t_vectors[float64_t_count];

class TmpVectorFloat64 {
private:
    int m_index;

public:
    TmpVectorFloat64() : m_index(-1) {
        for (int index = 0; index < float64_t_count; ++index) {
            if (!g_float64_t_used[index]) {
                g_float64_t_used[index] = true;
                m_index = index;
                return;
            }
        }
        assert(false);
    }

    ~TmpVectorFloat64() {
        g_float64_t_vectors[m_index].clear();
        g_float64_t_used[m_index] = false;
    }

    std::vector<float64_t>& vector(size_t size = 0) {
        g_float64_t_vectors[m_index].resize(size);
        return g_float64_t_vectors[m_index];
    }

    ArraySlice<float64_t> array_slice(const char* const name, size_t size = 0) {
        return ArraySlice<float64_t>(vector(size), name);
    }
};

template<typename D, typename O>
static void
downsample_slice(ConstArraySlice<D> input,
                 ArraySlice<O> output,
                 const size_t samples,
                 const size_t random_seed) {
    FastAssertCompare(samples, >=, 0);
    FastAssertCompare(output.size(), ==, input.size());

    if (input.size() == 0) {
        return;
    }

    if (input.size() == 1) {
        output[0] = O(float64_t(samples) < float64_t(input[0]) ? samples : input[0]);
        return;
    }

    TmpVectorSizeT raii_tree;
    auto tree = raii_tree.array_slice("tmp_tree", downsample_tmp_size(input.size()));
    initialize_tree(input, tree);
    size_t& total = tree[tree.size() - 1];

    if (total <= samples) {
        if (static_cast<const void*>(output.begin()) != static_cast<const void*>(input.begin())) {
            std::copy(input.begin(), input.end(), output.begin());
        }
        return;
    }

    std::fill(output.begin(), output.end(), O(0));

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
                 const size_t random_seed) {
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
                  const size_t random_seed) {
    WithoutGil without_gil{};

    ConstMatrixSlice<D> input{ input_matrix, "input_matrix" };
    MatrixSlice<O> output{ output_array, "output_array" };

    parallel_loop(input.rows_count(), [&](const size_t row_index) {
        size_t slice_seed = random_seed == 0 ? 0 : random_seed + row_index * 997;
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
                const size_t random_seed) {
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
                      const size_t random_seed) {
    WithoutGil without_gil{};

    ConstArraySlice<D> input_data{ input_data_array, "input_data_array" };
    ConstArraySlice<P> input_indptr{ input_indptr_array, "input_indptr_array" };
    ArraySlice<O> output{ output_array, "output_array" };

    parallel_loop(input_indptr.size() - 1, [&](size_t band_index) {
        size_t band_seed = random_seed == 0 ? 0 : random_seed + band_index * 997;
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

        output_indices[output_element_offset] = I(output_element_index);
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

        output_indices[output_element_offset] = I(output_element_index);
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

    TmpVectorSizeT raii_positions;
    auto tmp_positions = raii_positions.array_slice("tmp_positions", band_indices.size());

    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", band_indices.size());

    TmpVectorFloat64 raii_values;
    auto tmp_values = raii_values.array_slice("tmp_values", band_indices.size());

    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);
    std::sort(tmp_positions.begin(),
              tmp_positions.end(),
              [&](const size_t left_position, const size_t right_position) {
                  auto left_index = band_indices[left_position];
                  auto right_index = band_indices[right_position];
                  return left_index < right_index;
              });

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    const size_t tmp_size = tmp_positions.size();
    for (size_t location = 0; location < tmp_size; ++location) {
        size_t position = tmp_positions[location];
        tmp_indices[location] = band_indices[position];
        tmp_values[location] = float64_t(band_data[position]);
    }

    std::copy(tmp_indices.begin(), tmp_indices.end(), band_indices.begin());
    std::copy(tmp_values.begin(), tmp_values.end(), band_data.begin());
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
collect_top_row(const size_t row_index,
                const size_t degree,
                ConstMatrixSlice<float32_t>& similarity_matrix,
                ArraySlice<int32_t> output_indices,
                ArraySlice<float32_t> output_data,
                bool ranks) {
    const size_t columns_count = similarity_matrix.columns_count();
    const auto row_similarities = similarity_matrix.get_row(row_index);

    const size_t start_position = row_index * degree;
    const size_t stop_position = start_position + degree;

    TmpVectorSizeT raii_positions;
    auto tmp_positions = raii_positions.array_slice("tmp_positions", columns_count);
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);

    std::nth_element(tmp_positions.begin(),
                     tmp_positions.begin() + degree,
                     tmp_positions.end(),
                     [&](const size_t left_column_index, const size_t right_column_index) {
                         float32_t left_similarity = row_similarities[left_column_index];
                         float32_t right_similarity = row_similarities[right_column_index];
                         return left_similarity > right_similarity;
                     });

    auto row_data = output_data.slice(start_position, stop_position);
    auto row_indices = output_indices.slice(start_position, stop_position);
    std::copy(tmp_positions.begin(), tmp_positions.begin() + degree, row_indices.begin());
    std::sort(row_indices.begin(), row_indices.end());

    if (!ranks) {
#ifdef __INTEL_COMPILER
#    pragma simd
#endif
        for (size_t location = 0; location < degree; ++location) {
            size_t index = row_indices[location];
            row_data[location] = row_similarities[index];
        }

        return;
    }

    tmp_positions = tmp_positions.slice(0, degree);
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
        row_data[position] = float32_t(location + 1);
    }
}

/// See the Python `metacell.utilities.computation.top_per` function.
static void
collect_top(const size_t degree,
            const pybind11::array_t<float32_t>& input_similarity_matrix,
            pybind11::array_t<int32_t>& output_indices_array,
            pybind11::array_t<float32_t>& output_data_array,
            bool ranks) {
    WithoutGil without_gil{};

    ConstMatrixSlice<float32_t> similarity_matrix(input_similarity_matrix, "similarity_matrix");
    const size_t rows_count = similarity_matrix.rows_count();
    const size_t columns_count = similarity_matrix.columns_count();

    ArraySlice<int32_t> output_indices(output_indices_array, "output_indices");
    ArraySlice<float32_t> output_data(output_data_array, "output_data");

    FastAssertCompare(0, <, degree);
    FastAssertCompare(degree, <, columns_count);

    FastAssertCompare(output_indices.size(), ==, degree * rows_count);
    FastAssertCompare(output_data.size(), ==, degree * rows_count);

    parallel_loop(rows_count, [&](size_t row_index) {
        collect_top_row(row_index, degree, similarity_matrix, output_indices, output_data, ranks);
    });
}

static void
prune_band(const size_t band_index,
           const size_t pruned_degree,
           ConstCompressedMatrix<float32_t, int32_t, int64_t>& input_pruned_values,
           ArraySlice<float32_t> output_pruned_values,
           ArraySlice<int32_t> output_pruned_indices,
           ConstArraySlice<int64_t> output_pruned_indptr) {
    const auto start_position = output_pruned_indptr[band_index];
    const auto stop_position = output_pruned_indptr[band_index + 1];

    auto output_indices = output_pruned_indices.slice(start_position, stop_position);
    auto output_values = output_pruned_values.slice(start_position, stop_position);

    const auto input_indices = input_pruned_values.get_band_indices(band_index);
    const auto input_values = input_pruned_values.get_band_data(band_index);
    FastAssertCompare(input_indices.size(), ==, input_values.size());
    FastAssertCompare(input_values.size(), ==, input_values.size());

    if (input_values.size() <= pruned_degree) {
        std::copy(input_indices.begin(), input_indices.end(), output_indices.begin());
        std::copy(input_values.begin(), input_values.end(), output_values.begin());
        return;
    }

    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", input_values.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);
    std::nth_element(tmp_indices.begin(),
                     tmp_indices.begin() + pruned_degree,
                     tmp_indices.end(),
                     [&](const size_t left_column_index, const size_t right_column_index) {
                         const auto left_similarity = input_values[left_column_index];
                         const auto right_similarity = input_values[right_column_index];
                         return left_similarity > right_similarity;
                     });

    tmp_indices = tmp_indices.slice(0, pruned_degree);
    std::sort(tmp_indices.begin(), tmp_indices.end());

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t location = 0; location < pruned_degree; ++location) {
        size_t position = tmp_indices[location];
        output_indices[location] = input_indices[position];
        output_values[location] = input_values[position];
    }
}

/// See the Python `metacell.utilities.computation.prune_per` function.
static void
collect_pruned(const size_t pruned_degree,
               const pybind11::array_t<float32_t>& input_pruned_values_data,
               const pybind11::array_t<int32_t>& input_pruned_values_indices,
               const pybind11::array_t<int64_t>& input_pruned_values_indptr,
               pybind11::array_t<float32_t>& output_pruned_values_array,
               pybind11::array_t<int32_t>& output_pruned_indices_array,
               pybind11::array_t<int64_t>& output_pruned_indptr_array) {
    WithoutGil without_gil{};

    size_t size = input_pruned_values_indptr.size() - 1;
    ConstCompressedMatrix<float32_t, int32_t, int64_t> input_pruned_values(
        ConstArraySlice<float32_t>(input_pruned_values_data, "input_pruned_values_data"),
        ConstArraySlice<int32_t>(input_pruned_values_indices, "input_pruned_values_indices"),
        ConstArraySlice<int64_t>(input_pruned_values_indptr, "pruned_values_indptr"),
        int32_t(size),
        "pruned_values");

    ArraySlice<float32_t> output_pruned_values(output_pruned_values_array, "output_pruned_values");
    ArraySlice<int32_t> output_pruned_indices(output_pruned_indices_array, "output_pruned_indices");
    ArraySlice<int64_t> output_pruned_indptr(output_pruned_indptr_array, "output_pruned_indptr");

    FastAssertCompare(output_pruned_values.size(), >=, size * pruned_degree);
    FastAssertCompare(output_pruned_indices.size(), >=, size * pruned_degree);
    FastAssertCompare(output_pruned_indptr.size(), ==, size + 1);

    size_t start_position = output_pruned_indptr[0] = 0;
    for (size_t band_index = 0; band_index < size; ++band_index) {
        FastAssertCompare(start_position, ==, output_pruned_indptr[band_index]);
        auto input_values = input_pruned_values.get_band_data(band_index);
        if (input_values.size() <= pruned_degree) {
            start_position += input_values.size();
        } else {
            start_position += pruned_degree;
        }
        output_pruned_indptr[band_index + 1] = start_position;
    }

    parallel_loop(size, [&](size_t band_index) {
        prune_band(band_index,
                   pruned_degree,
                   input_pruned_values,
                   output_pruned_values,
                   output_pruned_indices,
                   output_pruned_indptr);
    });
}

template<typename D, typename I, typename P>
static void
shuffle_band(const size_t band_index, CompressedMatrix<D, I, P>& matrix, const size_t random_seed) {
    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", matrix.elements_count());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

    std::minstd_rand random(random_seed);
    std::shuffle(tmp_indices.begin(), tmp_indices.end(), random);

    auto band_indices = matrix.get_band_indices(band_index);
    tmp_indices = tmp_indices.slice(0, band_indices.size());
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
                   const size_t random_seed) {
    CompressedMatrix<D, I, P> matrix(ArraySlice<D>(data_array, "data"),
                                     ArraySlice<I>(indices_array, "indices"),
                                     ArraySlice<P>(indptr_array, "indptr"),
                                     elements_count,
                                     "compressed");

    parallel_loop(matrix.bands_count(), [&](size_t band_index) {
        size_t band_seed = random_seed == 0 ? 0 : random_seed + band_index * 997;
        shuffle_band(band_index, matrix, band_seed);
    });
}

template<typename D>
static void
shuffle_row(const size_t row_index, MatrixSlice<D>& matrix, const size_t random_seed) {
    std::minstd_rand random(random_seed);
    auto row = matrix.get_row(row_index);
    std::shuffle(row.begin(), row.end(), random);
}

/// See the Python `metacell.utilities.computation.shuffle_matrix` function.
template<typename D>
static void
shuffle_matrix(pybind11::array_t<D>& matrix_array, const size_t random_seed) {
    MatrixSlice<D> matrix(matrix_array, "matrix");

    parallel_loop(matrix.rows_count(), [&](size_t row_index) {
        size_t row_seed = random_seed == 0 ? 0 : random_seed + row_index * 997;
        shuffle_row(row_index, matrix, row_seed);
    });
}

template<typename D>
static D
rank_row_element(const size_t row_index, ConstMatrixSlice<D>& input, const size_t rank) {
    const auto row_input = input.get_row(row_index);
    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", input.columns_count());
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
rank_rows(const pybind11::array_t<D>& input_matrix,
          pybind11::array_t<D>& output_array,
          const size_t rank) {
    ConstMatrixSlice<D> input(input_matrix, "input");
    ArraySlice<D> output(output_array, "array");

    const size_t rows_count = input.rows_count();
    FastAssertCompare(rows_count, ==, output_array.size());
    FastAssertCompare(rank, <, input.columns_count());

    parallel_loop(rows_count, [&](size_t row_index) {
        output[row_index] = rank_row_element(row_index, input, rank);
    });
}

template<typename D>
static void
rank_matrix_row(const size_t row_index, MatrixSlice<D>& matrix, bool ascending) {
    auto row = matrix.get_row(row_index);
    size_t columns_count = matrix.columns_count();

    TmpVectorSizeT raii_positions;
    auto tmp_positions = raii_positions.array_slice("tmp_positions", columns_count);

    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", columns_count);

    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);
    if (ascending) {
        std::sort(tmp_positions.begin(),
                  tmp_positions.end(),
                  [&](const size_t left_column_index, const size_t right_column_index) {
                      const auto left_value = row[left_column_index];
                      const auto right_value = row[right_column_index];
                      return left_value < right_value;
                  });
    } else {
        std::sort(tmp_positions.begin(),
                  tmp_positions.end(),
                  [&](const size_t left_column_index, const size_t right_column_index) {
                      const auto left_value = row[left_column_index];
                      const auto right_value = row[right_column_index];
                      return left_value > right_value;
                  });
    }

    for (size_t column_index = 0; column_index < columns_count; ++column_index) {
        tmp_indices[tmp_positions[column_index]] = column_index;
    }

    for (size_t column_index = 0; column_index < columns_count; ++column_index) {
        row[column_index] = D(tmp_indices[column_index] + 1);
    }
}

/// See the Python `metacell.utilities.computation.rank_matrix_by_layout` function.
template<typename D>
static void
rank_matrix(pybind11::array_t<D>& array, const bool ascending) {
    MatrixSlice<D> matrix(array, "matrix");

    const size_t rows_count = matrix.rows_count();

    parallel_loop(rows_count,
                  [&](size_t row_index) { rank_matrix_row(row_index, matrix, ascending); });
}

/// See the Python `metacell.tools.outlier_cells._collect_fold_factors` function.
template<typename D>
static void
fold_factor_dense(pybind11::array_t<D>& data_array,
                  const float64_t min_gene_fold_factor,
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
            value = D(log((value + 1.0) / (expected + 1.0)) * LOG2_SCALE);
            if (value < min_gene_fold_factor) {
                value = 0;
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
                       const float64_t min_gene_fold_factor,
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
            value = D(log((value + 1.0) / (expected + 1.0)) * LOG2_SCALE);
            if (value < min_gene_fold_factor) {
                value = 0;
            }
        }
    });
}

static void
collect_distinct_abs_folds(ArraySlice<int32_t> gene_indices,
                           ArraySlice<float32_t> gene_folds,
                           ConstArraySlice<float64_t> fold_in_cell) {
    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", fold_in_cell.size());
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
        gene_indices[position] = int32_t(gene_index);
        gene_folds[position] = float32_t(fold_in_cell[gene_index]);
    }
}

static void
collect_distinct_high_folds(ArraySlice<int32_t> gene_indices,
                            ArraySlice<float32_t> gene_folds,
                            ConstArraySlice<float64_t> fold_in_cell) {
    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", fold_in_cell.size());
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
        gene_indices[position] = int32_t(gene_index);
        gene_folds[position] = float32_t(fold_in_cell[gene_index]);
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
auroc_data(std::vector<float64_t>& in_values, std::vector<float64_t>& out_values) {
    std::sort(in_values.rbegin(), in_values.rend());
    std::sort(out_values.rbegin(), out_values.rend());

    const size_t in_size = in_values.size();
    const size_t out_size = out_values.size();

    if (in_size == 0) {
        FastAssertCompare(out_size, >, 0);
        return 0.0;
    }

    if (out_size == 0) {
        FastAssertCompare(out_size, >, 0);
        return 1.0;
    }

    const float64_t in_scale = 1.0 / in_size;
    const float64_t out_scale = 1.0 / out_size;

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

    const bool is_all_in = in_count == in_size;
    const bool is_all_out = out_count == out_size;
    FastAssertCompare((is_all_in || is_all_out), ==, true);

    area += (out_size - out_count) * out_scale * (in_count + in_size) * in_scale / 2;

    return area;
}

template<typename D>
static void
auroc_dense_vector(const ConstArraySlice<D>& values,
                   const ConstArraySlice<bool>& labels,
                   const ConstArraySlice<float32_t>& scales,
                   const float64_t normalization,
                   float64_t* fold,
                   float64_t* auroc) {
    const size_t size = labels.size();
    FastAssertCompare(values.size(), ==, size);

    TmpVectorFloat64 raii_in_values;
    auto tmp_in_values = raii_in_values.vector();

    TmpVectorFloat64 raii_out_values;
    auto tmp_out_values = raii_out_values.vector();

    tmp_in_values.reserve(size);
    tmp_out_values.reserve(size);

    float64_t sum_in = 0.0;
    float64_t sum_out = 0.0;

    for (size_t index = 0; index < size; ++index) {
        const auto value = values[index] / scales[index];
        if (labels[index]) {
            tmp_in_values.push_back(value);
            sum_in += value;
        } else {
            tmp_out_values.push_back(value);
            sum_out += value;
        }
    }

    FastAssertCompare(tmp_in_values.size() + tmp_out_values.size(), ==, size);

    size_t num_in = tmp_in_values.size();
    size_t num_out = tmp_out_values.size();
    num_in += !num_in;
    num_out += !num_out;
    *fold = (sum_in / num_in + normalization) / (sum_out / num_out + normalization);
    *auroc = auroc_data(tmp_in_values, tmp_out_values);
}

template<typename D>
static void
auroc_dense_matrix(const pybind11::array_t<D>& values_array,
                   const pybind11::array_t<bool>& column_labels_array,
                   const pybind11::array_t<float32_t>& column_scales_array,
                   const float64_t normalization,
                   pybind11::array_t<float64_t>& folds_array,
                   pybind11::array_t<float64_t>& aurocs_array) {
    ConstMatrixSlice<D> values(values_array, "values");
    ConstArraySlice<bool> column_labels(column_labels_array, "column_labels");
    ConstArraySlice<float32_t> column_scales(column_scales_array, "column_scales");
    ArraySlice<float64_t> row_folds(folds_array, "row_folds");
    ArraySlice<float64_t> row_aurocs(aurocs_array, "row_aurocs");
    FastAssertCompare(normalization, >, 0);

    const size_t columns_count = values.columns_count();
    const size_t rows_count = values.rows_count();

    FastAssertCompare(column_labels.size(), ==, columns_count);
    FastAssertCompare(row_aurocs.size(), ==, rows_count);

    parallel_loop(rows_count, [&](size_t row_index) {
        auroc_dense_vector(values.get_row(row_index),
                           column_labels,
                           column_scales,
                           normalization,
                           &row_folds[row_index],
                           &row_aurocs[row_index]);
    });
}

template<typename D, typename I>
static void
auroc_compressed_vector(const ConstArraySlice<D>& values,
                        const ConstArraySlice<I>& indices,
                        const ConstArraySlice<bool>& labels,
                        const ConstArraySlice<float32_t>& scales,
                        const float64_t normalization,
                        float64_t* fold,
                        float64_t* auroc) {
    const size_t size = labels.size();
    const size_t nnz_count = values.size();
    FastAssertCompare(nnz_count, <=, size);

    TmpVectorFloat64 raii_in_values;
    auto tmp_in_values = raii_in_values.vector();

    TmpVectorFloat64 raii_out_values;
    auto tmp_out_values = raii_out_values.vector();

    tmp_in_values.reserve(size);
    tmp_out_values.reserve(size);

    float64_t sum_in = 0.0;
    float64_t sum_out = 0.0;

    size_t prev_index = 0;
    for (size_t position = 0; position < nnz_count; ++position) {
        size_t index = size_t(indices[position]);
        auto value = values[position] / scales[index];

        SlowAssertCompare(prev_index, <=, index);
        while (prev_index < index) {
            if (labels[prev_index]) {
                tmp_in_values.push_back(0.0);
            } else {
                tmp_out_values.push_back(0.0);
            }
            ++prev_index;
        }

        SlowAssertCompare(prev_index, ==, index);
        if (labels[index]) {
            tmp_in_values.push_back(value);
            sum_in += value;
        } else {
            tmp_out_values.push_back(value);
            sum_out += value;
        }
        ++prev_index;
    }

    FastAssertCompare(prev_index, <=, size);
    while (prev_index < size) {
        if (labels[prev_index]) {
            tmp_in_values.push_back(0.0);
        } else {
            tmp_out_values.push_back(0.0);
        }
        ++prev_index;
    }

    FastAssertCompare(prev_index, ==, size);
    FastAssertCompare(tmp_in_values.size() + tmp_out_values.size(), ==, size);

    size_t num_in = tmp_in_values.size();
    size_t num_out = tmp_out_values.size();
    num_in += !num_in;
    num_out += !num_out;
    *fold = (sum_in / num_in + normalization) / (sum_out / num_out + normalization);
    *auroc = auroc_data(tmp_in_values, tmp_out_values);
}

template<typename D, typename I, typename P>
static void
auroc_compressed_matrix(const pybind11::array_t<D>& values_data_array,
                        const pybind11::array_t<I>& values_indices_array,
                        const pybind11::array_t<P>& values_indptr_array,
                        size_t elements_count,
                        const pybind11::array_t<bool>& element_labels_array,
                        const pybind11::array_t<float32_t>& element_scales_array,
                        float64_t normalization,
                        pybind11::array_t<float64_t>& band_folds_array,
                        pybind11::array_t<float64_t>& band_aurocs_array) {
    ConstCompressedMatrix<D, I, P> values(ConstArraySlice<D>(values_data_array, "values_data"),
                                          ConstArraySlice<I>(values_indices_array,
                                                             "values_indices"),
                                          ConstArraySlice<P>(values_indptr_array, "values_indptr"),
                                          elements_count,
                                          "values");
    ConstArraySlice<bool> element_labels(element_labels_array, "element_labels");
    ConstArraySlice<float32_t> element_scales(element_scales_array, "element_scales");
    ArraySlice<float64_t> band_folds(band_folds_array, "band_folds");
    ArraySlice<float64_t> band_aurocs(band_aurocs_array, "band_aurocs");

    parallel_loop(values.bands_count(), [&](size_t band_index) {
        auroc_compressed_vector(values.get_band_data(band_index),
                                values.get_band_indices(band_index),
                                element_labels,
                                element_scales,
                                normalization,
                                &band_folds[band_index],
                                &band_aurocs[band_index]);
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
        float32_t logistic = logistics_dense_vectors(values.get_row(some_index),
                                                     values.get_row(other_index),
                                                     location,
                                                     scale);
        logistics.get_row(some_index)[other_index] = logistic;
        logistics.get_row(other_index)[some_index] = logistic;
    });
}

template<typename D>
static void
logistics_dense_matrices(const pybind11::array_t<D>& rows_values_array,
                         const pybind11::array_t<D>& columns_values_array,
                         pybind11::array_t<float32_t>& logistics_array,
                         const float64_t location,
                         const float64_t scale) {
    ConstMatrixSlice<D> rows_values(rows_values_array, "rows_values");
    ConstMatrixSlice<D> columns_values(columns_values_array, "columns_values");
    MatrixSlice<float32_t> logistics(logistics_array, "logistics");

    const size_t rows_count = rows_values.rows_count();
    const size_t columns_count = columns_values.rows_count();

    FastAssertCompare(logistics.rows_count(), ==, rows_count);
    FastAssertCompare(logistics.columns_count(), ==, columns_count);

    const size_t iterations_count = rows_count * columns_count;

    parallel_loop(iterations_count, [&](size_t iteration_index) {
        size_t row_index = iterations_count / columns_count;
        size_t column_index = iterations_count % columns_count;

        float32_t logistic = logistics_dense_vectors(rows_values.get_row(row_index),
                                                     columns_values.get_row(column_index),
                                                     location,
                                                     scale);
        logistics.get_row(row_index)[column_index] = logistic;
    });
}

static float64_t
cover_diameter(size_t points_count, float64_t area, float64_t cover_fraction) {
    float64_t point_area = area * cover_fraction / points_count;
    return sqrt(point_area) * 4 / M_PI;
}

static const size_t DELTAS_COUNT = 6;

static const ssize_t DELTAS[2][DELTAS_COUNT][2] = {
    { { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 0 } },
    { { -1, -1 }, { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 1 }, { -1, 0 } }
};

static size_t
distance(size_t left_x_index, size_t left_y_index, size_t right_x_index, size_t right_y_index) {
    ssize_t x_distance = ssize_t(left_x_index) - ssize_t(right_x_index);
    ssize_t y_distance = ssize_t(left_y_index) - ssize_t(right_y_index);
    return size_t(x_distance * x_distance + y_distance * y_distance);
}

template<typename D>
static void
cover_coordinates(const pybind11::array_t<D>& raw_x_coordinates_array,
                  const pybind11::array_t<D>& raw_y_coordinates_array,
                  pybind11::array_t<D>& spaced_x_coordinates_array,
                  pybind11::array_t<D>& spaced_y_coordinates_array,
                  const float64_t cover_fraction,
                  const float64_t noise_fraction,
                  const size_t random_seed) {
    FastAssertCompare(cover_fraction, >, 0);
    FastAssertCompare(cover_fraction, <, 1);
    FastAssertCompare(noise_fraction, >=, 0);

    ConstArraySlice<D> raw_x_coordinates(raw_x_coordinates_array, "raw_x_coordinates");
    ConstArraySlice<D> raw_y_coordinates(raw_y_coordinates_array, "raw_y_coordinates");
    ArraySlice<D> spaced_x_coordinates(spaced_x_coordinates_array, "spaced_x_coordinates");
    ArraySlice<D> spaced_y_coordinates(spaced_y_coordinates_array, "spaced_y_coordinates");

    size_t points_count = raw_x_coordinates.size();
    FastAssertCompare(points_count, >, 2);
    FastAssertCompare(raw_y_coordinates.size(), ==, points_count);
    FastAssertCompare(spaced_x_coordinates.size(), ==, points_count);
    FastAssertCompare(spaced_y_coordinates.size(), ==, points_count);

    std::minstd_rand random(random_seed);
    std::normal_distribution<float64_t> noise(0.0, noise_fraction);

    const auto x_min = *std::min_element(raw_x_coordinates.begin(), raw_x_coordinates.end());
    const auto y_min = *std::min_element(raw_y_coordinates.begin(), raw_y_coordinates.end());
    const auto x_max = *std::max_element(raw_x_coordinates.begin(), raw_x_coordinates.end());
    const auto y_max = *std::max_element(raw_y_coordinates.begin(), raw_y_coordinates.end());
    const auto x_size = x_max - x_min;
    const auto y_size = y_max - y_min;
    FastAssertCompare(x_size, >, 0);
    FastAssertCompare(y_size, >, 0);

    const auto point_diameter =
        cover_diameter(points_count, float64_t(x_size) * float64_t(y_size), cover_fraction);

    const auto x_step = point_diameter;
    const auto y_step = point_diameter * sqrt(3.0) / 2.0;

    const size_t x_layout_grid_size = 2 + size_t((x_max - x_min) / x_step);
    const size_t y_layout_grid_size = 2 + size_t((y_max - y_min) / y_step);

    std::vector<std::vector<ssize_t>> point_index_of_location;
    for (size_t x_index = 0; x_index < x_layout_grid_size; ++x_index) {
        point_index_of_location.emplace_back(y_layout_grid_size, -1);
    }

    std::vector<std::array<size_t, 2>> preferred_location_of_points(points_count);
    std::vector<std::array<size_t, 2>> location_of_points(points_count);

    std::vector<size_t> delta_indices(DELTAS_COUNT);
    std::iota(delta_indices.begin(), delta_indices.end(), 0);

    for (size_t each_point_index = 0; each_point_index < points_count; ++each_point_index) {
        ssize_t point_index = ssize_t(each_point_index);
        size_t y_index = size_t(round((raw_y_coordinates[point_index] - y_min) / y_step));
        size_t x_index;
        if (y_index % 2 == 0) {
            x_index = size_t(round((raw_x_coordinates[point_index] - x_min) / x_step));
        } else {
            x_index = size_t(round((raw_x_coordinates[point_index] - x_min) / x_step + 0.5));
        }
        preferred_location_of_points[point_index] = { x_index, y_index };

        while (point_index >= 0) {
            ssize_t other_point_index = point_index_of_location[x_index][y_index];
            if (other_point_index >= 0) {
                std::shuffle(delta_indices.begin(), delta_indices.end(), random);
                for (auto delta_index : delta_indices) {
                    auto delta_x = DELTAS[y_index % 2][delta_index][0];
                    auto delta_y = DELTAS[y_index % 2][delta_index][1];

                    auto other_x_index = x_index + delta_x;
                    auto other_y_index = y_index + delta_y;
                    if (other_x_index < 0 || x_layout_grid_size <= other_x_index
                        || other_y_index < 0 || y_layout_grid_size <= other_y_index) {
                        continue;
                    }

                    other_point_index = point_index_of_location[other_x_index][other_y_index];
                    if (other_point_index < 0) {
                        x_index = other_x_index;
                        y_index = other_y_index;
                        break;
                    }
                }
            }

            if (other_point_index >= 0) {
                std::shuffle(delta_indices.begin(), delta_indices.end(), random);
                for (auto delta_index : delta_indices) {
                    auto delta_x = DELTAS[y_index % 2][delta_index][0];
                    auto delta_y = DELTAS[y_index % 2][delta_index][1];

                    auto other_x_index = x_index + delta_x;
                    auto other_y_index = y_index + delta_y;
                    if (other_x_index < 0 || x_layout_grid_size <= other_x_index
                        || other_y_index < 0 || y_layout_grid_size <= other_y_index) {
                        continue;
                    }

                    other_point_index = point_index_of_location[other_x_index][other_y_index];
                    x_index = other_x_index;
                    y_index = other_y_index;
                    break;
                }
            }

            point_index_of_location[x_index][y_index] = point_index;
            location_of_points[point_index] = { x_index, y_index };
            point_index = other_point_index;
        }
    }

    auto verify_indices = [&]() {
        for (size_t x_index = 0; x_index < x_layout_grid_size; ++x_index) {
            for (size_t y_index = 0; y_index < y_layout_grid_size; ++y_index) {
                const auto point_index = point_index_of_location[x_index][y_index];
                if (point_index >= 0) {
                    const auto point_x_index = location_of_points[point_index][0];
                    const auto point_y_index = location_of_points[point_index][1];
                    FastAssertCompare(point_x_index, ==, x_index);
                    FastAssertCompare(point_y_index, ==, y_index);
                }
            }
        }

        for (size_t point_index = 0; point_index < points_count; ++point_index) {
            const auto x_index = location_of_points[point_index][0];
            const auto y_index = location_of_points[point_index][1];
            const auto location_point_index = point_index_of_location[x_index][y_index];
            FastAssertCompare(location_point_index, ==, point_index);
        }
    };

    verify_indices();

    for (size_t point_index = 0; point_index < points_count; ++point_index) {
        auto x_index = location_of_points[point_index][0];
        auto y_index = location_of_points[point_index][1];

        auto preferred_x_index = preferred_location_of_points[point_index][0];
        auto preferred_y_index = preferred_location_of_points[point_index][1];

        bool did_move = true;
        while (did_move) {
            did_move = false;
            auto current_distance =
                distance(x_index, y_index, preferred_x_index, preferred_y_index);

            std::shuffle(delta_indices.begin(), delta_indices.end(), random);
            for (auto delta_index : delta_indices) {
                auto delta_x = DELTAS[y_index % 2][delta_index][0];
                auto delta_y = DELTAS[y_index % 2][delta_index][1];

                auto near_x_index = x_index + delta_x;
                auto near_y_index = y_index + delta_y;
                if (near_x_index < 0 || x_layout_grid_size <= near_x_index || near_y_index < 0
                    || y_layout_grid_size <= near_y_index) {
                    continue;
                }

                auto near_distance =
                    distance(near_x_index, near_y_index, preferred_x_index, preferred_y_index);
                if (near_distance > current_distance) {
                    continue;
                }

                auto other_point_index = point_index_of_location[near_x_index][near_y_index];
                if (other_point_index < 0) {
                    continue;
                }

                auto other_preferred_x_index = preferred_location_of_points[other_point_index][0];
                auto other_preferred_y_index = preferred_location_of_points[other_point_index][1];

                auto other_current_distance =
                    distance(x_index, y_index, other_preferred_x_index, other_preferred_y_index);
                auto other_near_distance = distance(near_x_index,
                                                    near_y_index,
                                                    other_preferred_x_index,
                                                    other_preferred_y_index);
                if (other_current_distance > other_near_distance
                    || (near_distance == current_distance
                        && other_current_distance == other_near_distance)) {
                    continue;
                }

                point_index_of_location[x_index][y_index] = other_point_index;
                location_of_points[other_point_index] = { x_index, y_index };

                point_index_of_location[near_x_index][near_y_index] = point_index;
                location_of_points[point_index] = { near_x_index, near_y_index };

                x_index = near_x_index;
                y_index = near_y_index;
                did_move = true;
            }
        }
    }

    verify_indices();

    for (size_t point_index = 0; point_index < points_count; ++point_index) {
        const auto x_index = location_of_points[point_index][0];
        const auto y_index = location_of_points[point_index][1];
        spaced_y_coordinates[point_index] = D((y_index + noise(random)) * y_step + y_min);
        if (y_index % 2 == 0) {
            spaced_x_coordinates[point_index] = D((x_index + noise(random)) * x_step + x_min);
        } else {
            spaced_x_coordinates[point_index] = D((x_index + noise(random) - 0.5) * x_step + x_min);
        }
    }
}

static std::vector<std::vector<int32_t>>
collect_connected_nodes(ConstCompressedMatrix<float32_t, int32_t, int32_t>& incoming_weights,
                        ConstArraySlice<int32_t> seed_of_nodes) {
    const size_t nodes_count = seed_of_nodes.size();
    std::vector<std::vector<int32_t>> connected_nodes(nodes_count);

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        if (seed_of_nodes[node_index] >= 0) {
            continue;
        }
        auto node_incoming = incoming_weights.get_band_indices(node_index);
        auto& node_connected = connected_nodes[node_index];
        node_connected.resize(node_incoming.size());
        auto copied = std::copy_if(node_incoming.begin(),
                                   node_incoming.end(),
                                   node_connected.begin(),
                                   [&](int32_t other_node_index) {
                                       return seed_of_nodes[other_node_index] < 0;
                                   });
        node_connected.resize(copied - node_connected.begin());
    }

    return connected_nodes;
}

static size_t
choose_seed_node(const std::vector<size_t>& tmp_candidates,
                 const std::vector<std::vector<int32_t>>& connected_nodes,
                 const float32_t min_seed_size_quantile,
                 const float32_t max_seed_size_quantile,
                 std::minstd_rand& random) {
    size_t size = tmp_candidates.size();

    TmpVectorSizeT raii_positions;
    auto tmp_positions = raii_positions.array_slice("tmp_positions", size);
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);

    size_t min_seed_rank = size_t(floor((size - 1) * min_seed_size_quantile));
    size_t max_seed_rank = size_t(ceil((size - 1) * max_seed_size_quantile));
    FastAssertCompare(0, <=, min_seed_rank);
    FastAssertCompare(min_seed_rank, <=, max_seed_rank);
    FastAssertCompare(max_seed_rank, <=, size - 1);

    std::nth_element(tmp_positions.begin(),
                     tmp_positions.begin() + min_seed_rank,
                     tmp_positions.end(),
                     [&](const size_t left_position, const size_t right_position) {
                         const auto left_node_index = tmp_candidates[left_position];
                         const auto right_node_index = tmp_candidates[right_position];
                         const auto left_size = connected_nodes[left_node_index].size();
                         const auto right_size = connected_nodes[right_node_index].size();
                         return left_size < right_size;
                     });

    std::nth_element(tmp_positions.begin() + min_seed_rank,
                     tmp_positions.begin() + max_seed_rank,
                     tmp_positions.end(),
                     [&](const size_t left_position, const size_t right_position) {
                         const auto left_node_index = tmp_candidates[left_position];
                         const auto right_node_index = tmp_candidates[right_position];
                         const auto left_size = connected_nodes[left_node_index].size();
                         const auto right_size = connected_nodes[right_node_index].size();
                         return left_size < right_size;
                     });

    const size_t selected = random() % (max_seed_rank + 1 - min_seed_rank);
    const size_t position = tmp_positions[min_seed_rank + selected];
    size_t seed_node_index = tmp_candidates[position];

    LOCATED_LOG(false)                                           //
        << " node: " << seed_node_index                          //
        << " size: " << connected_nodes[seed_node_index].size()  //
        << std::endl;
    return seed_node_index;
}

template<typename T>
static void
remove_sorted(std::vector<T>& vector, T value) {
    auto position = std::lower_bound(vector.begin(), vector.end(), value);
    if (position != vector.end() && *position == value) {
        vector.erase(position);
    } else {
        LOCATED_LOG(true) << " OOPS! removing nonexistent value" << std::endl;
        assert(false);
    }
}

static void
store_seed_node(ConstCompressedMatrix<float32_t, int32_t, int32_t>& outgoing_weights,
                ConstCompressedMatrix<float32_t, int32_t, int32_t>& incoming_weights,
                const size_t seed_index,
                const size_t seed_node_index,
                std::vector<size_t>& tmp_candidates,
                std::vector<std::vector<int32_t>>& connected_nodes,
                ArraySlice<int32_t> seed_of_nodes,
                const size_t seed_size) {
    remove_sorted(tmp_candidates, seed_node_index);

    auto seed_incoming_nodes = incoming_weights.get_band_indices(seed_node_index);
    auto seed_incoming_weights = incoming_weights.get_band_data(seed_node_index);

    TmpVectorSizeT positions_raii;
    auto tmp_positions = positions_raii.vector(seed_incoming_nodes.size());
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);

    tmp_positions.erase(std::remove_if(tmp_positions.begin(),
                                       tmp_positions.end(),
                                       [&](size_t position) {
                                           return seed_of_nodes[seed_incoming_nodes[position]] >= 0;
                                       }),
                        tmp_positions.end());
    if (seed_size < tmp_positions.size()) {
        std::nth_element(tmp_positions.begin(),
                         tmp_positions.begin() + seed_size,
                         tmp_positions.end(),
                         [&](const size_t left_position, const size_t right_position) {
                             return seed_incoming_weights[left_position]
                                    > seed_incoming_weights[right_position];
                         });
        tmp_positions.resize(seed_size);
    }

    FastAssertCompare(seed_of_nodes[seed_node_index], <, 0);
    seed_of_nodes[seed_node_index] = int32_t(seed_index);
    connected_nodes[seed_node_index].clear();

    for (auto position : tmp_positions) {
        auto node_index = seed_incoming_nodes[position];
        connected_nodes[node_index].clear();
        seed_of_nodes[node_index] = int32_t(seed_index);
    }

    auto outgoing_nodes = outgoing_weights.get_band_indices(seed_node_index);
    for (size_t node_index : outgoing_nodes) {
        if (seed_of_nodes[node_index] < 0) {
            remove_sorted(connected_nodes[node_index], int32_t(seed_node_index));
        }
    }

    for (auto position : tmp_positions) {
        auto node_index = seed_incoming_nodes[position];
        auto outgoing_nodes = outgoing_weights.get_band_indices(node_index);
        for (size_t other_node_index : outgoing_nodes) {
            if (seed_of_nodes[other_node_index] < 0) {
                remove_sorted(connected_nodes[other_node_index], node_index);
            }
        }
    }
}

static bool
keep_large_candidates(std::vector<size_t>& tmp_candidates,
                      const std::vector<std::vector<int32_t>>& connected_nodes) {
    tmp_candidates.erase(std::remove_if(tmp_candidates.begin(),
                                        tmp_candidates.end(),
                                        [&](size_t candidate_node_index) {
                                            return connected_nodes[candidate_node_index].size()
                                                   == 0;
                                        }),
                         tmp_candidates.end());
    return tmp_candidates.size() > 0;
}

/// See the Python `metacell.tools.candidates._choose_seeds` function.
void
choose_seeds(const pybind11::array_t<float32_t>& outgoing_weights_data_array,
             const pybind11::array_t<int32_t>& outgoing_weights_indices_array,
             const pybind11::array_t<int32_t>& outgoing_weights_indptr_array,
             const pybind11::array_t<float32_t>& incoming_weights_data_array,
             const pybind11::array_t<int32_t>& incoming_weights_indices_array,
             const pybind11::array_t<int32_t>& incoming_weights_indptr_array,
             const size_t random_seed,
             const size_t max_seeds_count,
             const float32_t min_seed_size_quantile,
             const float32_t max_seed_size_quantile,
             pybind11::array_t<int32_t>& seed_of_nodes_array) {
    ArraySlice<int32_t> seed_of_nodes = ArraySlice<int32_t>(seed_of_nodes_array, "seed_of_nodes");
    size_t nodes_count = seed_of_nodes.size();
    FastAssertCompare(nodes_count, >, 0);

    ConstCompressedMatrix<float32_t, int32_t, int32_t> outgoing_weights(
        ConstArraySlice<float32_t>(outgoing_weights_data_array, "outgoing_weights_data"),
        ConstArraySlice<int32_t>(outgoing_weights_indices_array, "outgoing_weights_indices"),
        ConstArraySlice<int32_t>(outgoing_weights_indptr_array, "outgoing_weights_indptr"),
        int32_t(nodes_count),
        "outgoing_weights");
    FastAssertCompare(outgoing_weights.bands_count(), ==, nodes_count);

    ConstCompressedMatrix<float32_t, int32_t, int32_t> incoming_weights(
        ConstArraySlice<float32_t>(incoming_weights_data_array, "incoming_weights_data"),
        ConstArraySlice<int32_t>(incoming_weights_indices_array, "incoming_weights_indices"),
        ConstArraySlice<int32_t>(incoming_weights_indptr_array, "incoming_weights_indptr"),
        int32_t(nodes_count),
        "incoming_weights");
    FastAssertCompare(incoming_weights.bands_count(), ==, nodes_count);

    FastAssertCompare(0, <=, min_seed_size_quantile);
    FastAssertCompare(min_seed_size_quantile, <=, max_seed_size_quantile);
    FastAssertCompare(max_seed_size_quantile, <=, 1.0);

    std::vector<std::vector<int32_t>> connected_nodes =
        collect_connected_nodes(incoming_weights, seed_of_nodes);

    TmpVectorSizeT candidates_raii;
    auto tmp_candidates = candidates_raii.vector(nodes_count);
    std::iota(tmp_candidates.begin(), tmp_candidates.end(), 0);
    tmp_candidates.erase(std::remove_if(tmp_candidates.begin(),
                                        tmp_candidates.end(),
                                        [&](size_t candidate_node_index) {
                                            return seed_of_nodes[candidate_node_index] >= 0;
                                        }),
                         tmp_candidates.end());

    std::minstd_rand random(random_seed);
    size_t given_seeds_count =
        size_t(*std::max_element(seed_of_nodes.begin(), seed_of_nodes.end()) + 1);
    size_t seeds_count = given_seeds_count;

    FastAssertCompare(tmp_candidates.size(), >=, max_seeds_count - given_seeds_count);
    size_t mean_seed_size =
        size_t(ceil(tmp_candidates.size() / (max_seeds_count - given_seeds_count)));
    FastAssertCompare(mean_seed_size, >=, 1);

    while (seeds_count < max_seeds_count
           && keep_large_candidates(tmp_candidates, connected_nodes)) {
        size_t seed_node_index = choose_seed_node(tmp_candidates,
                                                  connected_nodes,
                                                  min_seed_size_quantile,
                                                  max_seed_size_quantile,
                                                  random);

        store_seed_node(outgoing_weights,
                        incoming_weights,
                        seeds_count,
                        seed_node_index,
                        tmp_candidates,
                        connected_nodes,
                        seed_of_nodes,
                        mean_seed_size);
        ++seeds_count;
    }

    size_t strong_seeds_count = seeds_count;
    FastAssertCompare(strong_seeds_count, >, given_seeds_count);

    if (seeds_count < max_seeds_count) {
        std::shuffle(tmp_candidates.begin(), tmp_candidates.end(), random);

        while (tmp_candidates.size() > 0 && seeds_count < max_seeds_count) {
            auto node_index = tmp_candidates.back();
            tmp_candidates.pop_back();
            seed_of_nodes[node_index] = int32_t(seeds_count);
            seeds_count += 1;
        }
    }

    if (seeds_count < max_seeds_count) {
        tmp_candidates.resize(nodes_count);
        std::iota(tmp_candidates.begin(), tmp_candidates.end(), 0);
        std::shuffle(tmp_candidates.begin(), tmp_candidates.end(), random);

        size_t failed_attempts = 0;
        while (seeds_count < max_seeds_count) {
            auto seed_index =
                random() % (strong_seeds_count - given_seeds_count) + given_seeds_count;
            bool did_find_first_node = false;

            failed_attempts += 1;
            for (auto node_index : tmp_candidates) {
                if (size_t(seed_of_nodes[node_index]) == seed_index) {
                    if (!did_find_first_node) {
                        did_find_first_node = true;
                    } else {
                        seed_of_nodes[node_index] = int32_t(seeds_count);
                        seeds_count += 1;
                        failed_attempts = 0;
                        break;
                    }
                }
            }

            FastAssertCompare(failed_attempts, <, 10 * (strong_seeds_count - given_seeds_count));
        }
    }

    FastAssertCompare(seeds_count, ==, max_seeds_count);
}

// Score information for one node for one partition.
struct NodeScore {
private:
    float64_t m_total_outgoing_weights;
    float64_t m_total_incoming_weights;
    float64_t m_score;

public:
    NodeScore()
      : m_total_outgoing_weights(0), m_total_incoming_weights(0), m_score(log2(EPSILON) / 2.0) {}

    void update_outgoing(const int direction, const float64_t edge_weight) {
        m_total_outgoing_weights += direction * edge_weight;
        SlowAssertCompare(m_total_outgoing_weights, >=, -EPSILON);
        m_total_outgoing_weights = std::max(m_total_outgoing_weights, 0.0);
        m_score = NaN;
    }

    void update_incoming(const int direction, const float64_t edge_weight) {
        m_total_incoming_weights += direction * edge_weight;
        SlowAssertCompare(m_total_incoming_weights, >=, -EPSILON);
        m_total_incoming_weights = std::max(m_total_incoming_weights, 0.0);
        m_score = NaN;
    }

    float64_t total_outgoing_weights() const { return m_total_outgoing_weights; }

    float64_t total_incoming_weights() const { return m_total_incoming_weights; }

    float64_t score() const {
        SlowAssertCompare(std::isnan(m_score), ==, false);
        return m_score;
    }

    float64_t rescore() {
        SlowAssertCompare(m_total_outgoing_weights, >=, 0);
        SlowAssertCompare(m_total_outgoing_weights, <=, 1 + EPSILON);
        SlowAssertCompare(m_total_incoming_weights, >=, 0);
        m_score = log2(EPSILON + m_total_outgoing_weights * m_total_incoming_weights) / 2.0;
        return m_score;
    }
};

static std::ostream&
operator<<(std::ostream& os, const NodeScore& node_score) {
    return os << node_score.score()
              << " total_outgoing_weights: " << node_score.total_outgoing_weights()
              << " total_incoming_weights: " << node_score.total_incoming_weights();
}

static std::vector<size_t>
initial_size_of_partitions(ConstArraySlice<int32_t> partitions_of_nodes) {
    std::vector<size_t> size_of_partitions;

    for (int32_t partition_index : partitions_of_nodes) {
        if (partition_index < 0) {
            continue;
        }
        if (size_t(partition_index) >= size_of_partitions.size()) {
            size_of_partitions.resize(partition_index + 1);
        }
        size_of_partitions[partition_index] += 1;
    }

    for (size_t partition_index = 0; partition_index < size_of_partitions.size();
         ++partition_index) {
        FastAssertCompare(size_of_partitions[partition_index], >, 0);
    }

    return size_of_partitions;
}

static float64_t
initial_incoming_scale(ConstCompressedMatrix<float32_t, int32_t, int32_t> incoming_weights) {
    size_t nodes_count = incoming_weights.bands_count();
    float64_t incoming_scale = 0;

    LOCATED_LOG(false) << " initial_incoming_scale_of_nodes" << std::endl;
    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        float64_t total_incoming_weights = 0;
        const auto& node_incoming_weights = incoming_weights.get_band_data(node_index);
        for (const auto incoming_weight : node_incoming_weights) {
            total_incoming_weights += incoming_weight;
        }
        LOCATED_LOG(false)                                            //
            << " node_index: " << node_index                          //
            << " total_incoming_weights: " << total_incoming_weights  //
            << std::endl;
        FastAssertCompare(total_incoming_weights, >, 0);
        incoming_scale += log2(total_incoming_weights);
    }

    return incoming_scale;
}

static std::vector<std::vector<NodeScore>>
initial_score_of_nodes_of_partitions(
    ConstCompressedMatrix<float32_t, int32_t, int32_t> outgoing_weights,
    ConstCompressedMatrix<float32_t, int32_t, int32_t> incoming_weights,
    ConstArraySlice<int32_t> partition_of_nodes,
    size_t partitions_count) {
    LOCATED_LOG(false) << " initial_score_of_partitions" << std::endl;
    size_t nodes_count = outgoing_weights.bands_count();

    std::vector<std::vector<NodeScore>> score_of_nodes_of_partitions(partitions_count);
    for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
        score_of_nodes_of_partitions[partition_index].resize(nodes_count);
    }

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        const int partition_index = partition_of_nodes[node_index];
#if ASSERT_LEVEL > 0
        float64_t total_outgoing_weights = 0;
#endif

        const auto& node_outgoing_indices = outgoing_weights.get_band_indices(node_index);
        const auto& node_outgoing_weights = outgoing_weights.get_band_data(node_index);
        for (size_t other_node_position = 0; other_node_position < node_outgoing_indices.size();
             ++other_node_position) {
            const int32_t other_node_index = node_outgoing_indices[other_node_position];
            const float32_t edge_weight = node_outgoing_weights[other_node_position];

            SlowAssertCompare(other_node_index, !=, node_index);
            SlowAssertCompare(edge_weight, >, 0);
            SlowAssertCompare(edge_weight, <, 1 + EPSILON);

#if ASSERT_LEVEL > 0
            total_outgoing_weights += edge_weight;
#endif

            const int other_partition_index = partition_of_nodes[other_node_index];

            LOCATED_LOG(false)                                       //
                << " from node_index: " << node_index                //
                << " from partition_index: " << partition_index      //
                << " to_node_index: " << other_node_index            //
                << " to_partition_index: " << other_partition_index  //
                << " edge weight: " << edge_weight                   //
                << std::endl;

            if (other_partition_index >= 0) {
                score_of_nodes_of_partitions[other_partition_index][node_index]
                    .update_outgoing(+1, edge_weight);
                score_of_nodes_of_partitions[other_partition_index][node_index].rescore();
            }

            if (partition_index >= 0) {
                score_of_nodes_of_partitions[partition_index][other_node_index]
                    .update_incoming(+1, edge_weight);
                score_of_nodes_of_partitions[partition_index][other_node_index].rescore();
            }
        }

#if ASSERT_LEVEL > 0
        FastAssertCompare(total_outgoing_weights, >, 1 - EPSILON);
        FastAssertCompare(total_outgoing_weights, <, 1 + EPSILON);
#endif
    }

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            if (int32_t(partition_index) == partition_of_nodes[node_index]) {
                LOCATED_LOG(false)                                                              //
                    << " node_index: " << node_index                                            //
                    << " current partition_index: " << partition_index                          //
                    << " score: " << score_of_nodes_of_partitions[partition_index][node_index]  //
                    << std::endl;
            } else {
                LOCATED_LOG(false)                                                              //
                    << " node_index: " << node_index                                            //
                    << " other partition_index: " << partition_index                            //
                    << " score: " << score_of_nodes_of_partitions[partition_index][node_index]  //
                    << std::endl;
            }
        }
    }

    return score_of_nodes_of_partitions;
}

static std::vector<float64_t>
initial_score_of_partitions(
    size_t nodes_count,
    ConstArraySlice<int32_t> partitions_of_nodes,
    const size_t partitions_count,
    const std::vector<std::vector<NodeScore>>& score_of_nodes_of_partitions) {
    std::vector<float64_t> score_of_partitions(partitions_count, 0);

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        const int partition_index = partitions_of_nodes[node_index];
        if (partition_index >= 0) {
            score_of_partitions[partition_index] +=
                score_of_nodes_of_partitions[partition_index][node_index].score();
        }
    }

    return score_of_partitions;
}

#if ASSERT_LEVEL > 0
std::function<void()> g_verify;
#endif

// Optimize the partition of a graph.
struct OptimizePartitions {
    ConstCompressedMatrix<float32_t, int32_t, int32_t> outgoing_weights;
    ConstCompressedMatrix<float32_t, int32_t, int32_t> incoming_weights;
    const size_t nodes_count;
    ArraySlice<int32_t> partition_of_nodes;
    std::vector<float64_t> temperature_of_nodes;
    std::vector<size_t> size_of_partitions;
    const size_t partitions_count;
    float64_t incoming_scale;
    std::vector<std::vector<NodeScore>> score_of_nodes_of_partitions;
    std::vector<float64_t> score_of_partitions;

    OptimizePartitions(const pybind11::array_t<float32_t>& outgoing_weights_array,
                       const pybind11::array_t<int32_t>& outgoing_indices_array,
                       const pybind11::array_t<int32_t>& outgoing_indptr_array,
                       const pybind11::array_t<float32_t>& incoming_weights_array,
                       const pybind11::array_t<int32_t>& incoming_indices_array,
                       const pybind11::array_t<int32_t>& incoming_indptr_array,
                       pybind11::array_t<int32_t>& partition_of_nodes_array)
      : outgoing_weights(ConstCompressedMatrix<float32_t, int32_t, int32_t>(
            ConstArraySlice<float32_t>(outgoing_weights_array, "outgoing_weights_array"),
            ConstArraySlice<int32_t>(outgoing_indices_array, "outgoing_indices_array"),
            ConstArraySlice<int32_t>(outgoing_indptr_array, "outgoing_indptr_array"),
            int32_t(outgoing_indptr_array.size() - 1),
            "outgoing_weights"))
      , incoming_weights(ConstCompressedMatrix<float32_t, int32_t, int32_t>(
            ConstArraySlice<float32_t>(incoming_weights_array, "incoming_weights_array"),
            ConstArraySlice<int32_t>(incoming_indices_array, "incoming_indices_array"),
            ConstArraySlice<int32_t>(incoming_indptr_array, "incoming_indptr_array"),
            int32_t(incoming_indptr_array.size() - 1),
            "incoming_weights"))
      , nodes_count(outgoing_weights.bands_count())
      , partition_of_nodes(partition_of_nodes_array, "partition_of_nodes")
      , temperature_of_nodes(nodes_count, 1.0)
      , size_of_partitions(initial_size_of_partitions(partition_of_nodes))
      , partitions_count(size_of_partitions.size())
      , incoming_scale(initial_incoming_scale(incoming_weights))
      , score_of_nodes_of_partitions(initial_score_of_nodes_of_partitions(outgoing_weights,
                                                                          incoming_weights,
                                                                          partition_of_nodes,
                                                                          partitions_count))
      , score_of_partitions(initial_score_of_partitions(nodes_count,
                                                        partition_of_nodes,
                                                        partitions_count,
                                                        score_of_nodes_of_partitions)) {}

    float64_t score(bool with_orphans = true) const {
        /*
        if (!with_orphans) {
            std::cerr << "node,partition,total_outgoing,total_incoming" << std::endl;
            for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
                for (size_t partition_index = 0; partition_index < partitions_count;
                     ++partition_index) {
                    std::cerr << node_index << ","       //
                              << partition_index << ","  //
                              << score_of_nodes_of_partitions[partition_index][node_index]
                                     .total_outgoing_weights()
                              << ","  //
                              << score_of_nodes_of_partitions[partition_index][node_index]
                                     .total_incoming_weights()  //
                              << std::endl;
                }
            }
        }
        */

        float64_t total_score = nodes_count * log2(float64_t(nodes_count)) - incoming_scale;
        size_t orphans_count = nodes_count;
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            const float64_t score_of_partition = score_of_partitions[partition_index];
            const size_t size_of_partition = size_of_partitions[partition_index];
            total_score +=
                score_of_partition - size_of_partition * log2(float64_t(size_of_partition));
            orphans_count -= size_of_partition;
        }
        if (with_orphans) {
            total_score += orphans_count * NodeScore().score();
            return total_score / nodes_count;
        } else {
            return total_score / (nodes_count - orphans_count);
        }
    }

#if ASSERT_LEVEL > 0
    void verify(const OptimizePartitions& other) {
#    define ASSERT_SAME(CONTEXT, FIELD, EPSILON)                                                 \
        if (fabs(double(this_##FIELD) - double(other_##FIELD)) > EPSILON) {                      \
            std::cerr << "OOPS! " << #CONTEXT << ": " << CONTEXT << " actual " << #FIELD << ": " \
                      << this_##FIELD << ": "                                                    \
                      << " computed " << #FIELD << ": " << other_##FIELD << ": " << std::endl;   \
            assert(false);                                                                       \
        } else

        ConstArraySlice<int32_t> other_partition_of_nodes = other.partition_of_nodes;
        for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
            auto this_partition_index = partition_of_nodes[node_index];
            auto other_partition_index = other_partition_of_nodes[node_index];
            ASSERT_SAME(node_index, partition_index, 0);
        }

        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
                const auto& this_score_of_node =
                    score_of_nodes_of_partitions[partition_index][node_index];
                const auto& other_score_of_node =
                    other.score_of_nodes_of_partitions[partition_index][node_index];

#    define ASSERT_SCORE_FIELD(FIELD)                                                    \
        if (fabs(this_score_of_node.FIELD() - other_score_of_node.FIELD()) >= EPSILON) { \
            std::cerr << "OOPS! partition_index: " << partition_index                    \
                      << " node_index: " << node_index << " actual " << #FIELD << ": "   \
                      << this_score_of_node.FIELD() << " computed " << #FIELD << " : "   \
                      << other_score_of_node.FIELD() << std::endl;                       \
            assert(false);                                                               \
        } else

                ASSERT_SCORE_FIELD(total_outgoing_weights);
                ASSERT_SCORE_FIELD(total_incoming_weights);
                ASSERT_SCORE_FIELD(score);
            }

            auto this_partition_score = score_of_partitions[partition_index];
            auto other_partition_score = other.score_of_partitions[partition_index];
            ASSERT_SAME(partition_index, partition_score, 1e-3);
        }

        auto this_score = score();
        auto other_score = other.score();
        LOCATED_LOG(false) << " score: " << this_score << " ~ " << other_score << std::endl;
        assert(fabs(this_score - other_score) < 1e-3);
    }
#endif

    void optimize(const size_t random_seed,
                  float64_t cooldown_pass,
                  float64_t cooldown_node,
                  int32_t cold_partitions,
                  float64_t cold_temperature) {
        std::minstd_rand random(random_seed);

        TmpVectorSizeT indices_raii;
        auto tmp_indices = indices_raii.vector(nodes_count);
        std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

        std::vector<uint8_t> frozen_nodes(nodes_count, uint8_t(1));
        size_t frozen_count = 0;
        for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
            auto partition_index = partition_of_nodes[node_index];
            if (0 <= partition_index && partition_index < int32_t(cold_partitions)) {
                temperature_of_nodes[node_index] = cold_temperature;
                frozen_nodes[node_index] = 1;
                frozen_count += 1;
            } else {
                temperature_of_nodes[node_index] = 1.0;
                frozen_nodes[node_index] = 0;
            }
            LOCATED_LOG(false)                                           //
                << " node: " << node_index                               //
                << " temperature: " << temperature_of_nodes[node_index]  //
                << std::endl;
        }

        TmpVectorSizeT partitions_raii;
        auto tmp_partitions = partitions_raii.vector(partitions_count);

        TmpVectorFloat64 partition_cold_diffs_raii;
        auto tmp_partition_cold_diffs = partition_cold_diffs_raii.vector(partitions_count);

        TmpVectorFloat64 partition_hot_diffs_raii;
        auto tmp_partition_hot_diffs = partition_hot_diffs_raii.vector(partitions_count);

        FastAssertCompare(cooldown_node, >=, 0.0);
        FastAssertCompare(cooldown_node, <=, 1.0);
        if (cooldown_pass != 0.0) {
            FastAssertCompare(cooldown_pass, >, 0.0);
            FastAssertCompare(cooldown_pass, <, 1.0);
        }
        float64_t cooldown_rate = 1.0 - cooldown_pass / nodes_count;
        float64_t temperature = 1.0;

        bool did_improve = true;
        bool did_skip = false;
        size_t total_skipped = 0;
        size_t total_improved = 0;
        size_t total_unimproved = 0;
        while (temperature > 0 || did_improve) {
            LOCATED_LOG(false)                          //
                << " temperature: " << temperature      //
                << " cooldown_rate: " << cooldown_rate  //
                << " score: " << score()                //
                << " did_improve: " << did_improve      //
                << " did_skip: " << did_skip            //
                << std::endl;
            if (did_improve) {
                did_improve = false;
                did_skip = false;
            } else {
                if (did_skip) {
                    did_skip = false;
                    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
                        if (!frozen_nodes[node_index]) {
                            float64_t node_temperature = temperature_of_nodes[node_index];
                            temperature = std::min(temperature, node_temperature);
                        }
                    }
                } else {
                    temperature = 0;
                }
                LOCATED_LOG(false)                           //
                    << " next_temperature: " << temperature  //
                    << std::endl;
            }

            std::shuffle(tmp_indices.begin(), tmp_indices.end(), random);
            size_t skipped = 0;
            size_t improved = 0;
            size_t unimproved = 0;
            for (size_t node_index : tmp_indices) {
                temperature *= cooldown_rate;
                LOCATED_LOG(false)                                           //
                    << " cooldown_rate: " << cooldown_rate                   //
                    << " temperature: " << temperature                       //
                    << " node: " << node_index                               //
                    << " temperature: " << temperature_of_nodes[node_index]  //
                    << std::endl;
                if (temperature_of_nodes[node_index] < temperature) {
                    did_skip = true;
                    ++skipped;
                    LOCATED_LOG(false)              //
                        << " node: " << node_index  //
                        << " skipped"               //
                        << std::endl;
                } else if (improve_node(node_index,
                                        tmp_partitions,
                                        tmp_partition_cold_diffs,
                                        tmp_partition_hot_diffs,
                                        random,
                                        temperature)) {
                    frozen_count -= frozen_nodes[node_index];
                    frozen_nodes[node_index] = uint8_t(0);
                    did_improve = true;
                    ++improved;
                    LOCATED_LOG(false)              //
                        << " node: " << node_index  //
                        << " improved"              //
                        << std::endl;
                } else {
                    frozen_count -= frozen_nodes[node_index];
                    frozen_nodes[node_index] = uint8_t(0);
                    temperature_of_nodes[node_index] = temperature * (1 - cooldown_node);
                    ++unimproved;
                    LOCATED_LOG(false)                                           //
                        << " node: " << node_index                               //
                        << " unimproved"                                         //
                        << " temperature: " << temperature_of_nodes[node_index]  //
                        << std::endl;
                }
            }
            LOCATED_LOG(false)                                      //
                << " improved: " << improved                        //
                << " unimproved: " << unimproved                    //
                << " skipped: " << skipped                          //
                << " total: " << (improved + unimproved + skipped)  //
                << " frozen: " << frozen_count                      //
                << std::endl;
            total_improved += improved;
            total_skipped += skipped;
            total_unimproved += unimproved;
        }

        LOCATED_LOG(false)                                                        //
            << " improved: " << total_improved                                    //
            << " unimproved: " << total_unimproved                                //
            << " skipped: " << total_skipped                                      //
            << " total: " << (total_improved + total_unimproved + total_skipped)  //
            << " frozen: " << frozen_count                                        //
            << std::endl;
        LOCATED_LOG(false)                          //
            << " temperature: " << temperature      //
            << " cooldown_rate: " << cooldown_rate  //
            << " score: " << score()                //
            << std::endl;
    }

    bool improve_node(size_t node_index,
                      std::vector<size_t>& tmp_partitions,
                      std::vector<float64_t>& tmp_partition_cold_diffs,
                      std::vector<float64_t>& tmp_partition_hot_diffs,
                      std::minstd_rand& random,
                      const float64_t temperature) {
        const auto current_partition_index = partition_of_nodes[node_index];
        LOCATED_LOG(false)                                              //
            << " node: " << node_index                                  //
            << " current_partition_index: " << current_partition_index  //
            << " size: "
            << (current_partition_index < 0 ? 0 : size_of_partitions[current_partition_index])  //
            << std::endl;

        if (current_partition_index >= 0 && size_of_partitions[current_partition_index] < 2) {
            return false;
        }

        collect_initial_partition_diffs(node_index,
                                        current_partition_index,
                                        tmp_partition_cold_diffs,
                                        tmp_partition_hot_diffs);

        collect_cold_partition_diffs(node_index, current_partition_index, tmp_partition_cold_diffs);

        const float64_t current_cold_diff = collect_candidate_partitions(current_partition_index,
                                                                         tmp_partition_cold_diffs,
                                                                         tmp_partition_hot_diffs,
                                                                         temperature,
                                                                         tmp_partitions);

        const int32_t chosen_partition_index =
            choose_target_partition(current_partition_index, random, tmp_partitions);
        if (chosen_partition_index < 0) {
            return false;
        }

        const float64_t chosen_cold_diff = tmp_partition_cold_diffs[chosen_partition_index];
        LOCATED_LOG(false)                                //
            << " chosen_cold_diff: " << chosen_cold_diff  //
            << std::endl;

        update_scores_of_nodes(node_index, current_partition_index, chosen_partition_index);

        update_partitions_of_nodes(node_index, current_partition_index, chosen_partition_index);

        update_sizes_of_partitions(current_partition_index, chosen_partition_index);

        update_scores_of_partition(current_partition_index,
                                   current_cold_diff,
                                   chosen_partition_index,
                                   chosen_cold_diff);

#if ASSERT_LEVEL > 0
        if (g_verify) {
            g_verify();
        }
#endif

        return true;
    }

    void collect_initial_partition_diffs(const size_t node_index,
                                         const int32_t current_partition_index,
                                         std::vector<float64_t>& tmp_partition_cold_diffs,
                                         std::vector<float64_t>& tmp_partition_hot_diffs) {
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            const int direction = 1 - 2 * (int32_t(partition_index) == current_partition_index);
            const auto score =
                direction * score_of_nodes_of_partitions[partition_index][node_index].score();
            tmp_partition_cold_diffs[partition_index] = score;
            tmp_partition_hot_diffs[partition_index] = score;
            LOCATED_LOG(false)                                                  //
                << " node_index: " << node_index                                //
                << " partition_index: " << partition_index                      //
                << " cold_diff: " << tmp_partition_cold_diffs[partition_index]  //
                << " hot_diff: " << tmp_partition_hot_diffs[partition_index]    //
                << std::endl;
        }
    }

    void collect_cold_partition_diffs(const size_t node_index,
                                      const int32_t current_partition_index,
                                      std::vector<float64_t>& tmp_partition_cold_diffs) {
        const auto& node_outgoing_indices = outgoing_weights.get_band_indices(node_index);
        const auto& node_incoming_indices = incoming_weights.get_band_indices(node_index);

        const auto& node_outgoing_weights = outgoing_weights.get_band_data(node_index);
        const auto& node_incoming_weights = incoming_weights.get_band_data(node_index);

        size_t outgoing_count = node_outgoing_indices.size();
        size_t incoming_count = node_incoming_indices.size();

        FastAssertCompare(outgoing_count, >, 0);
        FastAssertCompare(incoming_count, >, 0);

        size_t outgoing_position = 0;
        size_t incoming_position = 0;

        auto outgoing_node_index = node_outgoing_indices[outgoing_position];
        auto incoming_node_index = node_incoming_indices[incoming_position];

        auto outgoing_edge_weight = node_outgoing_weights[outgoing_position];
        auto incoming_edge_weight = node_incoming_weights[incoming_position];

        while (outgoing_position < outgoing_count || incoming_position < incoming_count) {
            const auto other_node_index = std::min(outgoing_node_index, incoming_node_index);
            const auto other_partition_index = partition_of_nodes[other_node_index];

            const int is_outgoing = int(outgoing_node_index == other_node_index);
            const int is_incoming = int(incoming_node_index == other_node_index);

            LOCATED_LOG(false)                                          //
                << " consider other_node " << other_node_index          //
                << " other_partition_index: " << other_partition_index  //
                << std::endl;

            if (other_partition_index >= 0) {
                NodeScore other_score =
                    score_of_nodes_of_partitions[other_partition_index][other_node_index];
                const float64_t old_score = other_score.score();

                LOCATED_LOG(false)                                          //
                    << " other_node_index: " << other_node_index            //
                    << " other_partition_index: " << other_partition_index  //
                    << " old score: " << other_score                        //
                    << std::endl;

                const int direction = 1 - 2 * (other_partition_index == current_partition_index);

                other_score.update_incoming(direction * is_outgoing, outgoing_edge_weight);
                LOCATED_LOG(false && is_outgoing)                         //
                    << " other_node_index: " << other_node_index          //
                    << " direction: " << direction                        //
                    << " outgoing_edge_weight: " << outgoing_edge_weight  //
                    << std::endl;

                other_score.update_outgoing(direction * is_incoming, incoming_edge_weight);
                LOCATED_LOG(false && is_incoming)                         //
                    << " other_node_index: " << other_node_index          //
                    << " direction: " << direction                        //
                    << " incoming_edge_weight: " << incoming_edge_weight  //
                    << std::endl;

                const float64_t new_score = other_score.rescore();
                LOCATED_LOG(false)                                          //
                    << " other_node_index: " << other_node_index            //
                    << " other_partition_index: " << other_partition_index  //
                    << " new score: " << other_score                        //
                    << std::endl;

                tmp_partition_cold_diffs[other_partition_index] += new_score - old_score;
            }

            outgoing_position += is_outgoing;
            incoming_position += is_incoming;

            if (outgoing_position < outgoing_count) {
                outgoing_node_index = node_outgoing_indices[outgoing_position];
                outgoing_edge_weight = node_outgoing_weights[outgoing_position];
            } else {
                outgoing_node_index = int32_t(nodes_count);
                outgoing_edge_weight = 0;
            }

            if (incoming_position < incoming_count) {
                incoming_node_index = node_incoming_indices[incoming_position];
                incoming_edge_weight = node_incoming_weights[incoming_position];
            } else {
                incoming_node_index = int32_t(nodes_count);
                incoming_edge_weight = 0;
            }
        }
    }

    float64_t collect_candidate_partitions(const int32_t current_partition_index,
                                           const std::vector<float64_t>& tmp_partition_cold_diffs,
                                           const std::vector<float64_t>& tmp_partition_hot_diffs,
                                           const float64_t temperature,
                                           std::vector<size_t>& tmp_partitions) {
        float64_t current_hot_diff = 0;
        float64_t current_cold_diff = 0;
        float64_t current_adjusted_cold_diff = 0;

        if (current_partition_index >= 0) {
            const float64_t hot_diff = tmp_partition_hot_diffs[current_partition_index];
            const size_t old_size = size_of_partitions[current_partition_index];
            const size_t new_size = old_size - 1;
            const float64_t old_score = score_of_partitions[current_partition_index];
            const float64_t old_adjusted_score = old_score - old_size * log2(float64_t(old_size));
            const float64_t cold_diff = tmp_partition_cold_diffs[current_partition_index];
            const float64_t new_score = old_score + cold_diff;
            const float64_t new_adjusted_score = new_score - new_size * log2(float64_t(new_size));
            const float64_t adjusted_cold_diff = new_adjusted_score - old_adjusted_score;
            LOCATED_LOG(false)                                              //
                << " current_partition_index: " << current_partition_index  //
                << " hot_diff: " << hot_diff                                //
                << " old_score: " << old_score                              //
                << " new_score: " << new_score                              //
                << " cold_diff: " << cold_diff                              //
                << " old_size: " << old_size                                //
                << " new_size: " << new_size                                //
                << " old_adjusted_score: " << old_adjusted_score            //
                << " new_adjusted_score: " << new_adjusted_score            //
                << " adjusted_cold_diff: " << adjusted_cold_diff            //
                << std::endl;
            current_cold_diff = cold_diff;
            current_hot_diff = hot_diff;
            current_adjusted_cold_diff = adjusted_cold_diff;
        }

        tmp_partitions.clear();
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            if (int32_t(partition_index) == current_partition_index) {
                continue;
            }
            const float64_t hot_diff = tmp_partition_hot_diffs[partition_index];
            const size_t old_size = size_of_partitions[partition_index];
            const size_t new_size = old_size + 1;
            const float64_t old_score = score_of_partitions[partition_index];
            const float64_t cold_diff = tmp_partition_cold_diffs[partition_index];
            const float64_t new_score = old_score + cold_diff;
            const float64_t old_adjusted_score = old_score - old_size * log2(float64_t(old_size));
            const float64_t new_adjusted_score = new_score - new_size * log2(float64_t(new_size));
            const float64_t adjusted_cold_diff = new_adjusted_score - old_adjusted_score;
            const float64_t total_diff =
                (current_hot_diff + hot_diff) * temperature
                + (current_adjusted_cold_diff + adjusted_cold_diff) * (1.0 - temperature);
            LOCATED_LOG(false)                                    //
                << " partition_index: " << partition_index        //
                << " old_score: " << old_score                    //
                << " new_score: " << new_score                    //
                << " hot_diff: " << hot_diff                      //
                << " cold_diff: " << cold_diff                    //
                << " old_size: " << old_size                      //
                << " new_size: " << new_size                      //
                << " old_adjusted_score: " << old_adjusted_score  //
                << " new_adjusted_score: " << new_adjusted_score  //
                << " adjusted_cold_diff: " << adjusted_cold_diff  //
                << " total_diff: " << total_diff                  //
                << std::endl;
            if (total_diff > 0) {
                tmp_partitions.push_back(partition_index);
            }
        }

        return current_cold_diff;
    }

    int32_t choose_target_partition(const int32_t current_partition_index,
                                    std::minstd_rand& random,
                                    const std::vector<size_t>& tmp_partitions) {
        int32_t chosen_partition_index = -1;
        if (tmp_partitions.size() == 0) {
            if (current_partition_index >= 0) {
                return -1;
            }
            chosen_partition_index = random() % partitions_count;
        } else {
            chosen_partition_index = int32_t(tmp_partitions[random() % tmp_partitions.size()]);
        }

        LOCATED_LOG(false)                                            //
            << " chosen_partition_index: " << chosen_partition_index  //
            << std::endl;

        return chosen_partition_index;
    }

    void update_scores_of_nodes(const size_t node_index,
                                const int32_t from_partition_index,
                                const int32_t to_partition_index) {
        auto score_of_nodes_of_from_partition =
            from_partition_index < 0 ? nullptr
                                     : &score_of_nodes_of_partitions[from_partition_index];
        auto& score_of_nodes_of_to_partition = score_of_nodes_of_partitions[to_partition_index];

        const auto& node_outgoing_indices = outgoing_weights.get_band_indices(node_index);
        const auto& node_incoming_indices = incoming_weights.get_band_indices(node_index);

        const auto& node_outgoing_weights = outgoing_weights.get_band_data(node_index);
        const auto& node_incoming_weights = incoming_weights.get_band_data(node_index);

        size_t outgoing_count = node_outgoing_indices.size();
        size_t incoming_count = node_incoming_indices.size();

        FastAssertCompare(outgoing_count, >, 0);
        FastAssertCompare(incoming_count, >, 0);

        size_t outgoing_position = 0;
        size_t incoming_position = 0;

        auto outgoing_node_index = node_outgoing_indices[outgoing_position];
        auto incoming_node_index = node_incoming_indices[incoming_position];

        auto outgoing_edge_weight = node_outgoing_weights[outgoing_position];
        auto incoming_edge_weight = node_incoming_weights[incoming_position];

        while (outgoing_position < outgoing_count || incoming_position < incoming_count) {
            const auto other_node_index = std::min(outgoing_node_index, incoming_node_index);

            const int is_outgoing = int(outgoing_node_index == other_node_index);
            const int is_incoming = int(incoming_node_index == other_node_index);

            if (score_of_nodes_of_from_partition) {
                auto& other_node_from_score = (*score_of_nodes_of_from_partition)[other_node_index];
                LOCATED_LOG(false)                                        //
                    << " other_node_index: " << other_node_index          //
                    << " from_partition_index: " << from_partition_index  //
                    << " old score: " << other_node_from_score            //
                    << std::endl;

                other_node_from_score.update_incoming(-is_outgoing, outgoing_edge_weight);
                other_node_from_score.update_outgoing(-is_incoming, incoming_edge_weight);
                other_node_from_score.rescore();

                LOCATED_LOG(false)                                        //
                    << " other_node_index: " << other_node_index          //
                    << " from_partition_index: " << from_partition_index  //
                    << " new score: " << other_node_from_score            //
                    << std::endl;
            }

            auto& other_node_to_score = score_of_nodes_of_to_partition[other_node_index];
            LOCATED_LOG(false)                                    //
                << " other_node_index: " << other_node_index      //
                << " to_partition_index: " << to_partition_index  //
                << " old score: " << other_node_to_score          //
                << std::endl;

            other_node_to_score.update_incoming(+is_outgoing, outgoing_edge_weight);
            other_node_to_score.update_outgoing(+is_incoming, incoming_edge_weight);
            other_node_to_score.rescore();

            LOCATED_LOG(false)                                    //
                << " other_node_index: " << other_node_index      //
                << " to_partition_index: " << to_partition_index  //
                << " new score: " << other_node_to_score          //
                << std::endl;

            outgoing_position += is_outgoing;
            incoming_position += is_incoming;

            if (outgoing_position < outgoing_count) {
                outgoing_node_index = node_outgoing_indices[outgoing_position];
                outgoing_edge_weight = node_outgoing_weights[outgoing_position];
            } else {
                outgoing_node_index = int32_t(nodes_count);
                outgoing_edge_weight = 0;
            }

            if (incoming_position < incoming_count) {
                incoming_node_index = node_incoming_indices[incoming_position];
                incoming_edge_weight = node_incoming_weights[incoming_position];
            } else {
                incoming_node_index = int32_t(nodes_count);
                incoming_edge_weight = 0;
            }
        }
    }

    void update_partitions_of_nodes(const size_t node_index,
                                    const int32_t from_partition_index,
                                    const int32_t to_partition_index) {
        SlowAssertCompare(partition_of_nodes[node_index], ==, from_partition_index);
        partition_of_nodes[node_index] = to_partition_index;
        LOCATED_LOG(false)                                        //
            << " set node_index: " << node_index                  //
            << " from_partition_index: " << from_partition_index  //
            << " to_partition_index: " << to_partition_index      //
            << std::endl;
    }

    void update_sizes_of_partitions(const int32_t from_partition_index,
                                    const int32_t to_partition_index) {
        if (from_partition_index >= 0) {
            SlowAssertCompare(size_of_partitions[from_partition_index], >, 1);
            size_of_partitions[from_partition_index] -= 1;
        }

        size_of_partitions[to_partition_index] += 1;
    }

    void update_scores_of_partition(size_t current_partition_index,
                                    float64_t current_cold_diff,
                                    size_t chosen_partition_index,
                                    float64_t chosen_cold_diff) {
        if (current_partition_index >= 0) {
            LOCATED_LOG(false)                                                     //
                << " from_partition_index: " << current_partition_index            //
                << " old score: " << score_of_partitions[current_partition_index]  //
                << " from_cold_diff: " << current_cold_diff                        //
                << std::endl;
            score_of_partitions[current_partition_index] += current_cold_diff;
            LOCATED_LOG(false)                                                     //
                << " from_partition_index: " << current_partition_index            //
                << " new score: " << score_of_partitions[current_partition_index]  //
                << std::endl;
        }

        LOCATED_LOG(false)                                                    //
            << " to_partition_index: " << chosen_partition_index              //
            << " old score: " << score_of_partitions[chosen_partition_index]  //
            << " to_cold_diff: " << chosen_cold_diff                          //
            << std::endl;
        score_of_partitions[chosen_partition_index] += chosen_cold_diff;
        LOCATED_LOG(false)                                                    //
            << " to_partition_index: " << chosen_partition_index              //
            << " new score: " << score_of_partitions[chosen_partition_index]  //
            << std::endl;
    }
};

static float64_t
optimize_partitions(const pybind11::array_t<float32_t>& outgoing_weights_array,
                    const pybind11::array_t<int32_t>& outgoing_indices_array,
                    const pybind11::array_t<int32_t>& outgoing_indptr_array,
                    const pybind11::array_t<float32_t>& incoming_weights_array,
                    const pybind11::array_t<int32_t>& incoming_indices_array,
                    const pybind11::array_t<int32_t>& incoming_indptr_array,
                    const unsigned int random_seed,
                    float64_t cooldown_pass,
                    float64_t cooldown_node,
                    pybind11::array_t<int32_t>& partition_of_nodes_array,
                    int32_t cold_partitions,
                    float64_t cold_temperature) {
    OptimizePartitions optimizer(outgoing_weights_array,
                                 outgoing_indices_array,
                                 outgoing_indptr_array,
                                 incoming_weights_array,
                                 incoming_indices_array,
                                 incoming_indptr_array,
                                 partition_of_nodes_array);

#if ASSERT_LEVEL > 1
    g_verify = [&]() {
        LOCATED_LOG(false) << " VERIFY" << std::endl;
        OptimizePartitions verifier(outgoing_weights_array,
                                    outgoing_indices_array,
                                    outgoing_indptr_array,
                                    incoming_weights_array,
                                    incoming_indices_array,
                                    incoming_indptr_array,
                                    partition_of_nodes_array);
        LOCATED_LOG(false) << " COMPARE" << std::endl;
        verifier.verify(optimizer);
        LOCATED_LOG(false) << " VERIFIED" << std::endl;
    };
#else
    g_verify = nullptr;
#endif

    optimizer.optimize(random_seed,
                       cooldown_pass,
                       cooldown_node,
                       cold_partitions,
                       cold_temperature);

    float64_t score = optimizer.score();
    return score;
}

static float64_t
score_partitions(const pybind11::array_t<float32_t>& outgoing_weights_array,
                 const pybind11::array_t<int32_t>& outgoing_indices_array,
                 const pybind11::array_t<int32_t>& outgoing_indptr_array,
                 const pybind11::array_t<float32_t>& incoming_weights_array,
                 const pybind11::array_t<int32_t>& incoming_indices_array,
                 const pybind11::array_t<int32_t>& incoming_indptr_array,
                 pybind11::array_t<int32_t>& partition_of_nodes_array,
                 bool with_orphans) {
    OptimizePartitions optimizer(outgoing_weights_array,
                                 outgoing_indices_array,
                                 outgoing_indptr_array,
                                 incoming_weights_array,
                                 incoming_indices_array,
                                 incoming_indptr_array,
                                 partition_of_nodes_array);
    return optimizer.score(with_orphans);
}

struct Sums {
    float64_t values;
    float64_t squared;
};

template<typename D>
static Sums
sum_row_values(ConstArraySlice<D> input_row) {
    const D* input_data = input_row.begin();
    const size_t columns_count = input_row.size();
    float64_t sum_values = 0;
    float64_t sum_squared = 0;
    for (size_t column_index = 0; column_index < columns_count; ++column_index) {
        const float64_t value = input_data[column_index];
        sum_values += value;
        sum_squared += value * value;
    }
    return Sums{ float64_t(sum_values), float64_t(sum_squared) };
}

template<typename D>
static float32_t
correlate_two_dense_rows(ConstArraySlice<D> some_values,
                         float64_t some_sum_values,
                         float64_t some_sum_squared,
                         ConstArraySlice<D> other_values,
                         float64_t other_sum_values,
                         float64_t other_sum_squared) {
    const size_t columns_count = some_values.size();
    const D* some_values_data = some_values.begin();
    const D* other_values_data = other_values.begin();
    float64_t both_sum_values = 0;

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t column_index = 0; column_index < columns_count; ++column_index) {
        both_sum_values +=
            float64_t(some_values_data[column_index]) * float64_t(other_values_data[column_index]);
    }

    float64_t correlation = columns_count * both_sum_values - some_sum_values * other_sum_values;
    float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
    float64_t other_factor =
        columns_count * other_sum_squared - other_sum_values * other_sum_values;
    float64_t both_factors = sqrt(some_factor * other_factor);
    if (both_factors != 0) {
        correlation /= both_factors;
        return std::max(std::min(float32_t(correlation), float32_t(1.0)), float32_t(-1.0));
    } else {
        return 0.0;
    }
}

struct SevenCorrelations {
    float64_t correlations[7];
};

template<typename D>
static SevenCorrelations
correlate_seven_dense_rows(ConstMatrixSlice<D> values,
                           const std::vector<float64_t>& row_sum_values,
                           const std::vector<float64_t>& row_sum_squared,
                           const size_t some_index,
                           const size_t other_begin_index) {
    const size_t columns_count = values.columns_count();

    const D* const some_values_data = values.get_row(some_index).begin();
    const D* const other_0_values_data = values.get_row(other_begin_index + 0).begin();
    const D* const other_1_values_data = values.get_row(other_begin_index + 1).begin();
    const D* const other_2_values_data = values.get_row(other_begin_index + 2).begin();
    const D* const other_3_values_data = values.get_row(other_begin_index + 3).begin();
    const D* const other_4_values_data = values.get_row(other_begin_index + 4).begin();
    const D* const other_5_values_data = values.get_row(other_begin_index + 5).begin();
    const D* const other_6_values_data = values.get_row(other_begin_index + 6).begin();

    float64_t both_0_sum_values = 0;
    float64_t both_1_sum_values = 0;
    float64_t both_2_sum_values = 0;
    float64_t both_3_sum_values = 0;
    float64_t both_4_sum_values = 0;
    float64_t both_5_sum_values = 0;
    float64_t both_6_sum_values = 0;

    // __asm__("int3");

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t column_index = 0; column_index < columns_count; ++column_index) {
        const float64_t some_value = float64_t(some_values_data[column_index]);
        both_0_sum_values += some_value * float64_t(other_0_values_data[column_index]);
        both_1_sum_values += some_value * float64_t(other_1_values_data[column_index]);
        both_2_sum_values += some_value * float64_t(other_2_values_data[column_index]);
        both_3_sum_values += some_value * float64_t(other_3_values_data[column_index]);
        both_4_sum_values += some_value * float64_t(other_4_values_data[column_index]);
        both_5_sum_values += some_value * float64_t(other_5_values_data[column_index]);
        both_6_sum_values += some_value * float64_t(other_6_values_data[column_index]);
    }

    const float64_t some_sum_values = row_sum_values[some_index];
    const float64_t other_0_sum_values = row_sum_values[other_begin_index + 0];
    const float64_t other_1_sum_values = row_sum_values[other_begin_index + 1];
    const float64_t other_2_sum_values = row_sum_values[other_begin_index + 2];
    const float64_t other_3_sum_values = row_sum_values[other_begin_index + 3];
    const float64_t other_4_sum_values = row_sum_values[other_begin_index + 4];
    const float64_t other_5_sum_values = row_sum_values[other_begin_index + 5];
    const float64_t other_6_sum_values = row_sum_values[other_begin_index + 6];

    const float64_t some_sum_squared = row_sum_squared[some_index];
    const float64_t other_0_sum_squared = row_sum_squared[other_begin_index + 0];
    const float64_t other_1_sum_squared = row_sum_squared[other_begin_index + 1];
    const float64_t other_2_sum_squared = row_sum_squared[other_begin_index + 2];
    const float64_t other_3_sum_squared = row_sum_squared[other_begin_index + 3];
    const float64_t other_4_sum_squared = row_sum_squared[other_begin_index + 4];
    const float64_t other_5_sum_squared = row_sum_squared[other_begin_index + 5];
    const float64_t other_6_sum_squared = row_sum_squared[other_begin_index + 6];

    SevenCorrelations results;
#define COLLECT_RESULTS(WHICH_OTHER)                                                              \
    {                                                                                             \
        results.correlations[WHICH_OTHER] = columns_count * both_##WHICH_OTHER##_sum_values       \
                                            - some_sum_values * other_##WHICH_OTHER##_sum_values; \
        const float64_t some_factor =                                                             \
            columns_count * some_sum_squared - some_sum_values * some_sum_values;                 \
        const float64_t other_factor =                                                            \
            columns_count * other_##WHICH_OTHER##_sum_squared                                     \
            - other_##WHICH_OTHER##_sum_values * other_##WHICH_OTHER##_sum_values;                \
        const float64_t both_factors = sqrt(some_factor * other_factor);                          \
        if (both_factors != 0) {                                                                  \
            results.correlations[WHICH_OTHER] /= both_factors;                                    \
            results.correlations[WHICH_OTHER] =                                                   \
                std::max(std::min(results.correlations[WHICH_OTHER], 1.0), -1.0);                 \
        } else {                                                                                  \
            results.correlations[WHICH_OTHER] = 0.0;                                              \
        }                                                                                         \
    }
    COLLECT_RESULTS(0)
    COLLECT_RESULTS(1)
    COLLECT_RESULTS(2)
    COLLECT_RESULTS(3)
    COLLECT_RESULTS(4)
    COLLECT_RESULTS(5)
    COLLECT_RESULTS(6)

    return results;
}

static size_t
unrolled_iterations_count(const size_t rows_count, const size_t unroll_size) {
    const size_t full_rows_groups_count = (rows_count - 1) / unroll_size;
    const size_t full_rows_groups_sum = (full_rows_groups_count * (full_rows_groups_count + 1)) / 2;
    const size_t last_rows_count = (rows_count - 1) % unroll_size;
    const size_t last_rows_size = size_t(ceil((rows_count - 1.0) / unroll_size));
    const size_t iterations_count =
        full_rows_groups_sum * unroll_size + last_rows_count * last_rows_size;
    return iterations_count;
}

template<typename D>
static void
correlate_dense(const pybind11::array_t<D>& input_array,
                pybind11::array_t<float32_t>& output_array) {
    ConstMatrixSlice<D> input(input_array, "input");
    MatrixSlice<float32_t> output(output_array, "output");

    auto rows_count = input.rows_count();

    FastAssertCompare(output.rows_count(), ==, input.rows_count());
    FastAssertCompare(output.columns_count(), ==, input.rows_count());

    TmpVectorFloat64 row_sum_values_raii;
    auto row_sum_values = row_sum_values_raii.vector(rows_count);

    TmpVectorFloat64 row_sum_squared_raii;
    auto row_sum_squared = row_sum_squared_raii.vector(rows_count);

    parallel_loop(rows_count, [&](size_t row_index) {
        auto sums = sum_row_values(input.get_row(row_index));
        row_sum_values[row_index] = sums.values;
        row_sum_squared[row_index] = sums.squared;
    });

    for (size_t entry_index = 0; entry_index < rows_count; ++entry_index) {
        output.get_row(entry_index)[entry_index] = 1.0;
    }

    const size_t unroll_size = 7;
    const size_t iterations_count = unrolled_iterations_count(rows_count, unroll_size);

    parallel_loop(iterations_count, [&](size_t iteration_index) {
        size_t min_rows_count = size_t(round(
            (sqrt(unroll_size * (unroll_size + 8.0 * iteration_index)) - unroll_size + 1.0) / 2.0));
        size_t min_rows_iterations_count = unrolled_iterations_count(min_rows_count, unroll_size);

        while (min_rows_count > 1 and min_rows_iterations_count > iteration_index) {
            min_rows_count -= 1;
            min_rows_iterations_count = unrolled_iterations_count(min_rows_count, unroll_size);
        }

        while (true) {
            const size_t up_min_rows_iterations_count =
                unrolled_iterations_count(min_rows_count + 1, unroll_size);
            if (up_min_rows_iterations_count > iteration_index) {
                break;
            }
            min_rows_count += 1;
            min_rows_iterations_count = up_min_rows_iterations_count;
        }

        const size_t some_index = min_rows_count;
        const size_t extra_iterations = iteration_index - min_rows_iterations_count;
        const size_t other_begin_index = extra_iterations * unroll_size;
        const size_t other_end_index = std::min(other_begin_index + unroll_size, some_index);

        if (other_begin_index + 7 == other_end_index) {
            const SevenCorrelations results = correlate_seven_dense_rows(input,
                                                                         row_sum_values,
                                                                         row_sum_squared,
                                                                         some_index,
                                                                         other_begin_index);
            for (int which_other = 0; which_other < 7; ++which_other) {
                const size_t other_index = other_begin_index + which_other;
                output.get_row(some_index)[other_index] = results.correlations[which_other];
                output.get_row(other_index)[some_index] = results.correlations[which_other];
            }
        } else {
            for (size_t other_index = other_begin_index; other_index != other_end_index;
                 ++other_index) {
                float32_t correlation = correlate_two_dense_rows(input.get_row(some_index),
                                                                 row_sum_values[some_index],
                                                                 row_sum_squared[some_index],
                                                                 input.get_row(other_index),
                                                                 row_sum_values[other_index],
                                                                 row_sum_squared[other_index]);
                output.get_row(some_index)[other_index] = correlation;
                output.get_row(other_index)[some_index] = correlation;
            }
        }
    });
}

template<typename D, typename I>
static float32_t
correlate_compressed_rows(const size_t columns_count,
                          ConstArraySlice<I> some_indices,
                          ConstArraySlice<D> some_values,
                          float64_t some_sum_values,
                          float64_t some_sum_squared,
                          ConstArraySlice<I> other_indices,
                          ConstArraySlice<D> other_values,
                          float64_t other_sum_values,
                          float64_t other_sum_squared) {
    float64_t both_sum_values = 0;
    const size_t some_count = some_indices.size();
    const size_t other_count = other_indices.size();
    size_t some_location = 0;
    size_t other_location = 0;
    while (some_location < some_count && other_location < other_count) {
        auto some_index = some_indices[some_location];
        auto other_index = other_indices[other_location];
        float64_t some_value = float64_t(some_values[some_location]);
        float64_t other_value = float64_t(other_values[other_location]);
        both_sum_values += some_value * other_value * (some_index == other_index);
        some_location += some_index <= other_index;
        other_location += other_index <= some_index;
    }

    float64_t correlation = columns_count * both_sum_values - some_sum_values * other_sum_values;
    float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
    float64_t other_factor =
        columns_count * other_sum_squared - other_sum_values * other_sum_values;
    float64_t both_factors = sqrt(some_factor * other_factor);
    if (both_factors != 0) {
        correlation /= both_factors;
    } else {
        correlation = 0;
    }
    return std::max(std::min(float32_t(correlation), float32_t(1.0)), float32_t(-1.0));
}

template<typename D, typename I, typename P>
static void
correlate_compressed(const pybind11::array_t<D>& input_data_array,
                     const pybind11::array_t<I>& input_indices_array,
                     const pybind11::array_t<P>& input_indptr_array,
                     size_t columns_count,
                     pybind11::array_t<float32_t>& output_array) {
    ConstCompressedMatrix<D, I, P> input(ConstArraySlice<D>(input_data_array, "input_data"),
                                         ConstArraySlice<I>(input_indices_array, "input_indices"),
                                         ConstArraySlice<P>(input_indptr_array, "input_indptr"),
                                         columns_count,
                                         "input");
    MatrixSlice<float32_t> output(output_array, "output");

    auto rows_count = input.bands_count();

    FastAssertCompare(output.rows_count(), ==, input.bands_count());
    FastAssertCompare(output.columns_count(), ==, input.bands_count());

    TmpVectorFloat64 row_sum_values_raii;
    auto row_sum_values = row_sum_values_raii.vector(rows_count);

    TmpVectorFloat64 row_sum_squared_raii;
    auto row_sum_squared = row_sum_squared_raii.vector(rows_count);

    parallel_loop(rows_count, [&](size_t row_index) {
        const auto sums = sum_row_values(input.get_band_data(row_index));
        row_sum_values[row_index] = sums.values;
        row_sum_squared[row_index] = sums.squared;
    });

    const size_t iterations_count = (rows_count * (rows_count - 1)) / 2;

    for (size_t entry_index = 0; entry_index < rows_count; ++entry_index) {
        output.get_row(entry_index)[entry_index] = 1.0;
    }

    parallel_loop(iterations_count, [&](size_t iteration_index) {
        size_t some_index = iteration_index / (rows_count - 1);
        size_t other_index = iteration_index % (rows_count - 1);
        if (other_index < rows_count - 1 - some_index) {
            some_index = rows_count - 1 - some_index;
        } else {
            other_index = rows_count - 2 - other_index;
        }
        const float32_t correlation = correlate_compressed_rows(columns_count,
                                                                input.get_band_indices(some_index),
                                                                input.get_band_data(some_index),
                                                                row_sum_values[some_index],
                                                                row_sum_squared[some_index],
                                                                input.get_band_indices(other_index),
                                                                input.get_band_data(other_index),
                                                                row_sum_values[other_index],
                                                                row_sum_squared[other_index]);
        output.get_row(some_index)[other_index] = correlation;
        output.get_row(other_index)[some_index] = correlation;
    });
}

}  // namespace metacells

PYBIND11_MODULE(extensions, module) {
    module.doc() = "C++ extensions to support the metacells package.";

    module.def("set_threads_count",
               &metacells::set_threads_count,
               "Specify the number of parallel threads.");
    module.def("cover_diameter",
               &metacells::cover_diameter,
               "The diameter for points to achieve plot area coverage.");
    module.def("choose_seeds",
               &metacells::choose_seeds,
               "Choose seed partitions for computing metacells.");
    module.def("optimize_partitions",
               &metacells::optimize_partitions,
               "Optimize the partition for computing metacells.");
    module.def("score_partitions",
               &metacells::score_partitions,
               "Compute the quality score for metacells.");

#define REGISTER_D(D)                                                                             \
    module.def("shuffle_matrix_" #D, &metacells::shuffle_matrix<D>, "Shuffle matrix data.");      \
    module.def("rank_rows_" #D,                                                                   \
               &metacells::rank_rows<D>,                                                          \
               "Collect the rank element in each row.");                                          \
    module.def("rank_matrix_" #D, &metacells::rank_matrix<D>, "Replace matrix data with ranks."); \
    module.def("fold_factor_dense_" #D,                                                           \
               &metacells::fold_factor_dense<D>,                                                  \
               "Fold factors of dense data.");                                                    \
    module.def("auroc_dense_matrix_" #D,                                                          \
               &metacells::auroc_dense_matrix<D>,                                                 \
               "AUROC for dense matrix.");                                                        \
    module.def("logistics_dense_matrix_" #D,                                                      \
               &metacells::logistics_dense_matrix<D>,                                             \
               "Logistics distances for dense matrix.");                                          \
    module.def("logistics_dense_matrices_" #D,                                                    \
               &metacells::logistics_dense_matrices<D>,                                           \
               "Logistics distances for dense matrices.");                                        \
    module.def("cover_coordinates_" #D,                                                           \
               &metacells::cover_coordinates<D>,                                                  \
               "Move points to achieve plot area coverage.");                                     \
    module.def("correlate_dense_" #D,                                                             \
               &metacells::correlate_dense<D>,                                                    \
               "Correlate rows of a dense matrix.");

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
               "AUROC for compressed matrix.");              \
    module.def("correlate_compressed_" #D "_" #I "_" #P,     \
               &metacells::correlate_compressed<D, I, P>,    \
               "Correlate rows of a compressed matrix.");

    module.def("collect_top", &metacells::collect_top, "Collect the topmost elements.");
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
