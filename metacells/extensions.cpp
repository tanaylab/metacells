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
    int m_size;          ///< Number of elements.
    const char* m_name;  ///< Name for error messages.

public:
    /// Construct an array slice from raw data.
    ConstArraySlice(const T* const data, const int size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    /// Construct an array slice based on a Python Numpy array.
    ConstArraySlice(const pybind11::array_t<T>& array, const char* const name)
      : ConstArraySlice(array.data(), array.size(), name) {
        FastAssertCompareWhat(array.ndim(), ==, 1, name);
        FastAssertCompareWhat(array.data(1) - array.data(0), ==, 1, name);
    }

    /// Split into two immutable sub-slices.
    template<typename I>
    std::pair<ConstArraySlice, ConstArraySlice> split(const I size) const {
        return std::make_pair(slice(0, size), slice(size, m_size));
    }

    /// Return an immutable sub-slice.
    template<typename I, typename J>
    ConstArraySlice slice(const I start, const J stop) const {
        FastAssertCompareWhat(0, <=, start, m_name);
        FastAssertCompareWhat(start, <=, stop, m_name);
        FastAssertCompareWhat(stop, <=, m_size, m_name);
        return ConstArraySlice(m_data + start, stop - start, m_name);
    }

    /// The number of elements in the slice.
    int size() const { return m_size; }

    /// Access immutable element by index.
    template<typename I>
    const T& operator[](const I index) const {
        SlowAssertCompareWhat(0, <=, index, m_name);
        SlowAssertCompareWhat(index, <, m_size, m_name);
        return m_data[index];
    }

    /// Beginning of immutable data for STD algorithms.
    const T* begin() const { return m_data; }

    /// End of immutable data for STD algorithms.
    const T* end() const { return m_data + m_size; }
};

/// A mutable contiguous slice of an array of type ``T``.
template<typename T>
class ArraySlice {
private:
    T* m_data;           ///< Pointer to the first element.
    int m_size;          ///< Number of elements.
    const char* m_name;  ///< Name for error messages.

public:
    /// Construct an array slice from raw data.
    ArraySlice(T* const data, const int size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    /// Construct an array slice based on a Python Numpy array.
    ArraySlice(pybind11::array_t<T>& array, const char* const name)
      : ArraySlice(array.mutable_data(), array.size(), name) {
        FastAssertCompareWhat(array.ndim(), ==, 1, name);
        FastAssertCompareWhat(array.data(1) - array.data(0), ==, 1, name);
    }

    /// Split into two mutable sub-slices.
    template<typename I>
    std::pair<ArraySlice, ArraySlice> split(const I size) {
        return std::make_pair(slice(0, size), slice(size, m_size));
    }

    /// Return a mutable sub-slice.
    template<typename I, typename J>
    ArraySlice slice(const I start, const J stop) {
        FastAssertCompareWhat(0, <=, start, m_name);
        FastAssertCompareWhat(start, <=, stop, m_name);
        FastAssertCompareWhat(stop, <=, m_size, m_name);
        return ArraySlice(m_data + start, stop - start, m_name);
    }

    /// The number of elements in the slice.
    int size() const { return m_size; }

    /// Access mutable element by index.
    template<typename I>
    T& operator[](const I index) {
        SlowAssertCompareWhat(0, <=, index, m_name);
        SlowAssertCompareWhat(index, <, m_size, m_name);
        return m_data[index];
    }

    /// Beginning of mutable data for STD algorithms.
    T* begin() { return m_data; }

    /// End of mutable data for STD algorithms.
    T* end() { return m_data + m_size; }

    /// Implicit conversion to an immutable slice.
    operator ConstArraySlice<T>() const { return ConstArraySlice<T>(m_data, m_size, m_name); }
};

/// An immutable row-major slice of a matrix of type ``T``.
template<typename T>
class ConstMatrixSlice {
private:
    const T* m_data;      ///< Pointer to the first element.
    int m_rows_count;     ///< Number of rows.
    int m_columns_count;  ///< Number of columns.
    int m_rows_offset;    ///< Offset between start of rows.
    const char* m_name;   ///< Name for error messages.

public:
    /// Construct a matrix slice from raw data.
    ConstMatrixSlice(const T* const data,
                     const int rows_count,
                     const int columns_count,
                     const int rows_offset,
                     const char* const name)
      : m_data(data)
      , m_rows_count(rows_count)
      , m_columns_count(columns_count)
      , m_rows_offset(rows_offset)
      , m_name(name) {}

    /// Construct a matrix slice based on a Python Numpy array.
    ConstMatrixSlice(const pybind11::array_t<T>& array, const char* const name)
      : ConstMatrixSlice(array.data(),
                         array.shape(0),
                         array.shape(1),
                         array.data(1, 0) - array.data(0, 0),
                         name) {
        FastAssertCompareWhat(array.ndim(), ==, 2, name);
        FastAssertCompareWhat(array.data(0, 1) - array.data(0, 0), ==, 1, name);
        FastAssertCompareWhat(m_columns_count, <=, m_rows_offset, name);
    }

    /// Obtain a specific row of the matrix slice as an array slice.
    template<typename I>
    ConstArraySlice<T> get_row(I row_index) const {
        FastAssertCompareWhat(0, <=, row_index, m_name);
        FastAssertCompareWhat(row_index, <, m_rows_count, m_name);
        return ConstArraySlice<T>(m_data + row_index * m_rows_offset, m_columns_count, m_name);
    }

    /// The number of rows in the matrix slice.
    int rows_count() const { return m_rows_count; }

    /// The number of columns in the matrix slice.
    int columns_count() const { return m_columns_count; }

    /// Access immutable element by its indices.
    template<typename I, typename J>
    const T& operator()(const I row_index, const J column_index) const {
        SlowAssertCompareWhat(0, <=, row_index, m_name);
        SlowAssertCompareWhat(row_index, <, m_rows_count, m_name);
        SlowAssertCompareWhat(0, <=, column_index, m_name);
        SlowAssertCompareWhat(column_index, <, m_columns_count, m_name);
        return m_data[row_index * m_rows_offset + column_index];
    }
};

/// A mutable row-major slice of a matrix of type ``T``.
template<typename T>
class MatrixSlice {
private:
    T* m_data;            ///< Pointer to the first element.
    int m_rows_count;     ///< Number of rows.
    int m_columns_count;  ///< Number of columns.
    int m_rows_offset;    ///< Offset between start of rows.
    const char* m_name;   ///< Name for error messages.

public:
    /// Construct a matrix slice from raw data.
    MatrixSlice(T* const data,
                const int rows_count,
                const int columns_count,
                const int rows_offset,
                const char* const name)
      : m_data(data)
      , m_rows_count(rows_count)
      , m_columns_count(columns_count)
      , m_rows_offset(rows_offset)
      , m_name(name) {}

    /// Construct a matrix slice based on a Python Numpy array.
    MatrixSlice(pybind11::array_t<T>& array, const char* const name)
      : MatrixSlice(array.mutable_data(),
                    array.shape(0),
                    array.shape(1),
                    array.data(1, 0) - array.data(0, 0),
                    name) {
        FastAssertCompareWhat(array.ndim(), ==, 2, name);
        FastAssertCompareWhat(array.data(0, 1) - array.data(0, 0), ==, 1, name);
        FastAssertCompareWhat(m_columns_count, <=, m_rows_offset, name);
    }

    /// Obtain a specific row of the matrix slice as an array slice.
    template<typename I>
    ArraySlice<T> get_row(I row_index) const {
        FastAssertCompareWhat(0, <=, row_index, m_name);
        FastAssertCompareWhat(row_index, <, m_rows_count, m_name);
        return ArraySlice<T>(m_data + row_index * m_rows_offset, m_columns_count, m_name);
    }

    /// The number of rows in the matrix slice.
    int rows_count() const { return m_rows_count; }

    /// The number of columns in the matrix slice.
    int columns_count() const { return m_columns_count; }

    /// Access mutable element by its indices.
    template<typename I, typename J>
    T& operator()(const I row_index, const J column_index) const {
        SlowAssertCompareWhat(0, <=, row_index, m_name);
        SlowAssertCompareWhat(row_index, <, m_rows_count, m_name);
        SlowAssertCompareWhat(0, <=, column_index, m_name);
        SlowAssertCompareWhat(column_index, <, m_columns_count, m_name);
        return m_data[row_index * m_rows_offset + column_index];
    }

    /// Implicit conversion to an immutable slice.
    operator ConstMatrixSlice<T>() const {
        return ConstMatrixSlice<T>(m_data, m_rows_count, m_columns_count, m_rows_offset, m_name);
    }
};

/// An immutable CSR sparse matrix.
template<typename P, typename I, typename D>
class ConstCsrMatrix {
private:
    ConstArraySlice<P> m_indptr;   ///< First and last indices positions per row.
    ConstArraySlice<I> m_indices;  ///< Column indices.
    ConstArraySlice<D> m_data;     ///< Non-zero data.
    int m_rows_count;              ///< Number of rows.
    int m_columns_count;           ///< Number of columns.
    const char* m_name;            ///< Name for error messages.

public:
    /// Construct a matrix slice based on a Python Numpy array.
    ConstCsrMatrix(ConstArraySlice<P>&& indptr,
                   ConstArraySlice<I>&& indices,
                   ConstArraySlice<D>&& data,
                   const I columns_count,
                   const char* const name)
      : m_indptr(indptr)
      , m_indices(indices)
      , m_data(data)
      , m_rows_count(indptr.size() - 1)
      , m_columns_count(columns_count)
      , m_name(name) {
        FastAssertCompareWhat(m_indptr[m_rows_count], ==, indices.size(), m_name);
        FastAssertCompareWhat(m_indptr[m_rows_count], ==, data.size(), m_name);
    }

    template<typename J>
    ConstArraySlice<I> get_row_indices(const J row_index) const {
        auto start_position = m_indptr[row_index];
        auto stop_position = m_indptr[row_index + 1];
        return m_indices.slice(start_position, stop_position);
    }

    template<typename J>
    ConstArraySlice<D> get_row_data(const J row_index) const {
        auto start_position = m_indptr[row_index];
        auto stop_position = m_indptr[row_index + 1];
        return m_data.slice(start_position, stop_position);
    }
};

/// A mutable CSR sparse matrix.
template<typename P, typename I, typename D>
class CsrMatrix {
private:
    ArraySlice<P> m_indptr;   ///< First and last indices positions per row.
    ArraySlice<I> m_indices;  ///< Column indices.
    ArraySlice<D> m_data;     ///< Non-zero data.
    int m_rows_count;         ///< Number of rows.
    int m_columns_count;      ///< Number of columns.
    const char* m_name;       ///< Name for error messages.

public:
    /// Construct a matrix slice based on a Python Numpy array.
    CsrMatrix(pybind11::array_t<P>& indptr_array,
              pybind11::array_t<I>& indices_array,
              pybind11::array_t<D>& data_array,
              const I columns_count,
              const char* const name)
      : m_indptr(indptr_array)
      , m_indices(indices_array)
      , m_data(data_array)
      , m_rows_count(indptr_array.size() - 1)
      , m_columns_count(columns_count)
      , m_name(name) {
        FastAssertCompareWhat(m_indptr[m_rows_count], ==, indices_array.size(), m_name);
        FastAssertCompareWhat(m_indptr[m_rows_count], ==, data_array.size(), m_name);
    }

    template<typename J>
    ArraySlice<I> get_row_indices(const J row_index) {
        auto start_position = m_indptr[row_index];
        auto stop_position = m_indptr[row_index + 1];
        return m_indices.slice(start_position, stop_position);
    }

    template<typename J>
    ArraySlice<D> get_row_data(const J row_index) {
        auto start_position = m_indptr[row_index];
        auto stop_position = m_indptr[row_index + 1];
        return m_data.slice(start_position, stop_position);
    }
};

template<typename T>
static T
ceil_power_of_two(const T size) {
    return 1 << int(ceil(log2(size)));
}

static int
downsample_tmp_size(const int size) {
    if (size <= 1) {
        return 0;
    }
    return 2 * ceil_power_of_two(size) - 1;
}

template<typename D>
static void
initialize_tree(ConstArraySlice<D> input, ArraySlice<int32_t> tree) {
    FastAssertCompare(input.size(), >=, 2);

    int input_size = ceil_power_of_two(input.size());
    std::copy(input.begin(), input.end(), tree.begin());
    std::fill(tree.begin() + input.size(), tree.begin() + input_size, 0);

    while (input_size > 1) {
        auto slices = tree.split(input_size);
        auto input = slices.first;
        tree = slices.second;

        input_size /= 2;
        for (int index = 0; index < input_size; ++index) {
            const auto left = input[index * 2];
            const auto right = input[index * 2 + 1];
            tree[index] = left + right;

            SlowAssertCompare(left, >=, 0);
            SlowAssertCompare(right, >=, 0);
            SlowAssertCompare(tree[index], ==, int64_t(left) + int64_t(right));
        }
    }
    FastAssertCompare(tree.size(), ==, 1);
}

static int
random_sample(ArraySlice<int32_t> tree, int64_t random) {
    int size_of_level = 1;
    int base_of_level = tree.size() - 1;
    int index_in_level = 0;
    int index_in_tree = base_of_level + index_in_level;

    while (true) {
        SlowAssertCompare(index_in_tree, ==, base_of_level + index_in_level);
        FastAssertCompare(tree[index_in_tree], >, random);
        --tree[base_of_level + index_in_level];
        size_of_level *= 2;
        base_of_level -= size_of_level;
        if (base_of_level < 0) {
            return index_in_level;
        }
        index_in_level *= 2;
        SlowAssertCompare(tree[base_of_level + index_in_level]
                              + tree[base_of_level + index_in_level + 1],
                          ==,
                          tree[base_of_level + size_of_level + index_in_level / 2] + 1);

        index_in_tree = base_of_level + index_in_level;
        int64_t right_random = random - int64_t(tree[index_in_tree]);
        if (right_random >= 0) {
            ++index_in_level;
            ++index_in_tree;
            SlowAssertCompare(index_in_level, <, size_of_level);
            random = right_random;
        }
    }
}

static thread_local std::vector<int32_t> tmp_positions;
static thread_local std::vector<int32_t> tmp_indices;
static thread_local std::vector<double> tmp_data;
static thread_local std::vector<int32_t> tmp_tree;

template<typename D, typename O>
static void
downsample_slice(ConstArraySlice<D> input,
                 ArraySlice<O> output,
                 const int samples,
                 const int random_seed) {
    FastAssertCompare(samples, >=, 0);
    FastAssertCompare(output.size(), ==, input.size());

    tmp_tree.resize(downsample_tmp_size(input.size()));
    ArraySlice<int32_t> tree(&tmp_tree[0], tmp_tree.size(), "tree");

    if (input.size() == 0) {
        return;
    }

    if (input.size() == 1) {
        output[0] = double(samples) < double(input[0]) ? samples : input[0];
        return;
    }

    initialize_tree(input, tree);
    auto& total = tree[tree.size() - 1];

    if (total <= samples) {
        if (static_cast<const void*>(output.begin()) != static_cast<const void*>(input.begin())) {
            std::copy(input.begin(), input.end(), output.begin());
        }
        return;
    }

    std::fill(output.begin(), output.end(), 0);

    std::minstd_rand random(random_seed);
    for (int index = 0; index < samples; ++index) {
        ++output[random_sample(tree, random() % int64_t(total))];
    }
}

/// See the Python `metacell.utilities.computation.downsample_array` function.
template<typename D, typename O>
static void
downsample_array(const pybind11::array_t<D>& input_array,
                 pybind11::array_t<O>& output_array,
                 const int samples,
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
                  const int samples,
                  const int random_seed) {
    WithoutGil without_gil{};

    ConstMatrixSlice<D> input{ input_matrix, "input_matrix" };
    MatrixSlice<O> output{ output_array, "output_array" };

    const int rows_count = input.rows_count();

#pragma omp parallel for
    for (int row_index = 0; row_index < rows_count; ++row_index) {
        downsample_slice(input.get_row(row_index), output.get_row(row_index), samples, random_seed);
    }
}

template<typename D, typename P, typename O>
static void
downsample_band(const int band_index,
                ConstArraySlice<D> input_data,
                ConstArraySlice<P> input_indptr,
                ArraySlice<O> output,
                const int samples,
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
                      const int samples,
                      const int random_seed) {
    WithoutGil without_gil{};

    ConstArraySlice<D> input_data{ input_data_array, "input_data_array" };
    ConstArraySlice<P> input_indptr{ input_indptr_array, "input_indptr_array" };
    ArraySlice<O> output{ output_array, "output_array" };

    const int bands_count = input_indptr.size() - 1;

#pragma omp parallel for
    for (int band_index = 0; band_index < bands_count; ++band_index) {
        downsample_band(band_index, input_data, input_indptr, output, samples, random_seed);
    }
}

template<typename D, typename I, typename P>
static void
collect_compressed_band(const int input_band_index,
                        ConstArraySlice<D> input_data,
                        ConstArraySlice<I> input_indices,
                        ConstArraySlice<P> input_indptr,
                        ArraySlice<D> output_data,
                        ArraySlice<I> output_indices,
                        ArraySlice<P> output_indptr) {
    auto start_input_element_offset = input_indptr[input_band_index];
    auto stop_input_element_offset = input_indptr[input_band_index + 1];

    FastAssertCompare(0, <=, start_input_element_offset);
    FastAssertCompare(start_input_element_offset, <=, stop_input_element_offset);
    FastAssertCompare(stop_input_element_offset, <=, input_data.size());

    int output_element_index = input_band_index;

    for (P input_element_offset = start_input_element_offset;
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

    const int input_bands_count = input_indptr.size() - 1;

#pragma omp parallel for
    for (int input_band_index = 0; input_band_index < input_bands_count; ++input_band_index) {
        collect_compressed_band(input_band_index,
                                input_data,
                                input_indices,
                                input_indptr,
                                output_data,
                                output_indices,
                                output_indptr);
    }
}

template<typename D, typename I, typename P>
static void
sort_band(const int band_index,
          ArraySlice<D> data,
          ArraySlice<I> indices,
          ConstArraySlice<P> indptr) {
    auto start_element_offset = indptr[band_index];
    auto stop_element_offset = indptr[band_index + 1];
    if (stop_element_offset == start_element_offset) {
        return;
    }

    auto band_indices = indices.slice(start_element_offset, stop_element_offset);
    auto band_data = data.slice(start_element_offset, stop_element_offset);

    tmp_positions.resize(stop_element_offset - start_element_offset);
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);
    std::sort(tmp_positions.begin(),
              tmp_positions.end(),
              [&](const int left_position, const int right_position) {
                  auto left_index = band_indices[left_position];
                  auto right_index = band_indices[right_position];
                  return left_index < right_index;
              });

    tmp_indices.resize(tmp_positions.size());
    tmp_data.resize(tmp_positions.size());

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    const int tmp_size = tmp_positions.size();
    for (int location = 0; location < tmp_size; ++location) {
        int32_t position = tmp_positions[location];
        tmp_indices[location] = band_indices[position];
        tmp_data[location] = band_data[position];
    }

    std::copy(tmp_indices.begin(), tmp_indices.end(), band_indices.begin());
    std::copy(tmp_data.begin(), tmp_data.end(), band_data.begin());
}

/// See the Python `metacell.utilities.computation._relayout_compressed` function.
template<typename D, typename I, typename P>
static void
sort_compressed(pybind11::array_t<D>& data_array,
                pybind11::array_t<I>& indices_array,
                pybind11::array_t<P>& indptr_array) {
    WithoutGil without_gil{};

    ArraySlice<D> data{ data_array, "data_array" };
    ArraySlice<I> indices{ indices_array, "indices_array" };
    ConstArraySlice<P> indptr{ indptr_array, "indptr_array" };

    FastAssertCompare(data.size(), ==, indptr[indptr.size() - 1]);
    FastAssertCompare(indices.size(), ==, data.size());

    const int bands_count = indptr.size() - 1;

#pragma omp parallel for
    for (int band_index = 0; band_index < bands_count; ++band_index) {
        sort_band(band_index, data, indices, indptr);
    }
}

static void
collect_outgoing_row(const int row_index,
                     const int degree,
                     ConstMatrixSlice<float32_t>& similarity_matrix,
                     ArraySlice<int32_t> output_indices,
                     ArraySlice<float32_t> output_ranks) {
    const int size = similarity_matrix.rows_count();
    const auto row_similarities = similarity_matrix.get_row(row_index);

    const int start_position = row_index * degree;
    const int stop_position = start_position + degree;

    auto row_indices = output_indices.slice(start_position, stop_position);
    auto row_ranks = output_ranks.slice(start_position, stop_position);

    if (degree < size - 1) {
        tmp_positions.resize(size - 1);
        std::iota(tmp_positions.begin(), tmp_positions.begin() + row_index, 0);
        std::iota(tmp_positions.begin() + row_index, tmp_positions.end(), row_index + 1);

        std::nth_element(tmp_positions.begin(),
                         tmp_positions.begin() + degree,
                         tmp_positions.end(),
                         [&](const int32_t left_column_index, const int32_t right_column_index) {
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
              [&](const int left_position, const int right_position) {
                  float32_t left_similarity = row_similarities[row_indices[left_position]];
                  float32_t right_similarity = row_similarities[row_indices[right_position]];
                  return left_similarity < right_similarity;
              });
#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (int location = 0; location < degree; ++location) {
        int position = tmp_positions[location];
        row_ranks[position] = location + 1;
    }
}

/// See the Python `metacell.tools.knn_graph._rank_outgoing` function.
static void
collect_outgoing(const int degree,
                 const pybind11::array_t<float32_t>& input_similarity_matrix,
                 pybind11::array_t<int32_t>& output_indices_array,
                 pybind11::array_t<float32_t>& output_ranks_array) {
    WithoutGil without_gil{};

    ConstMatrixSlice<float32_t> similarity_matrix(input_similarity_matrix, "similarity_matrix");
    FastAssertCompareWhat(similarity_matrix.rows_count(),
                          ==,
                          similarity_matrix.columns_count(),
                          "similarity_matrix");
    const int size = similarity_matrix.rows_count();

    ArraySlice<int32_t> output_indices(output_indices_array, "output_indices");
    ArraySlice<float32_t> output_ranks(output_ranks_array, "output_ranks");

    FastAssertCompare(0, <, degree);
    FastAssertCompare(degree, <, size);

    FastAssertCompare(output_indices.size(), ==, degree * size);
    FastAssertCompare(output_ranks.size(), ==, degree * size);

#pragma omp parallel for
    for (int row_index = 0; row_index < size; ++row_index) {
        collect_outgoing_row(row_index, degree, similarity_matrix, output_indices, output_ranks);
    }
}

static void
prune_row(const int row_index,
          const int pruned_degree,
          ConstCsrMatrix<int32_t, int32_t, float32_t>& input_pruned_ranks,
          ConstArraySlice<int32_t> output_pruned_indptr,
          ArraySlice<int32_t> output_pruned_indices,
          ArraySlice<float32_t> output_pruned_ranks) {
    const auto start_position = output_pruned_indptr[row_index];
    const auto stop_position = output_pruned_indptr[row_index + 1];

    auto output_indices = output_pruned_indices.slice(start_position, stop_position);
    auto output_ranks = output_pruned_ranks.slice(start_position, stop_position);

    const auto input_indices = input_pruned_ranks.get_row_indices(row_index);
    const auto input_ranks = input_pruned_ranks.get_row_data(row_index);
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
                     [&](const int32_t left_column_index, const int32_t right_column_index) {
                         const auto left_similarity = input_ranks[left_column_index];
                         const auto right_similarity = input_ranks[right_column_index];
                         return left_similarity > right_similarity;
                     });

    tmp_indices.resize(pruned_degree);
    std::sort(tmp_indices.begin(), tmp_indices.end());

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (int location = 0; location < pruned_degree; ++location) {
        int position = tmp_indices[location];
        output_indices[location] = input_indices[position];
        output_ranks[location] = input_ranks[position];
    }
}

/// See the Python `metacell.tools.knn_graph._prune_ranks` function.
static void
collect_pruned(const int pruned_degree,
               const pybind11::array_t<int32_t>& input_pruned_ranks_indptr,
               const pybind11::array_t<int32_t>& input_pruned_ranks_indices,
               const pybind11::array_t<float32_t>& input_pruned_ranks_data,
               pybind11::array_t<int32_t>& output_pruned_indptr_array,
               pybind11::array_t<int32_t>& output_pruned_indices_array,
               pybind11::array_t<float32_t>& output_pruned_ranks_array) {
    int size = input_pruned_ranks_indptr.size() - 1;
    ConstCsrMatrix<int32_t, int32_t, float32_t> input_pruned_ranks(
        ConstArraySlice<int32_t>(input_pruned_ranks_indptr, "pruned_ranks_indptr"),
        ConstArraySlice<int32_t>(input_pruned_ranks_indices, "input_pruned_ranks_indices"),
        ConstArraySlice<float32_t>(input_pruned_ranks_data, "input_pruned_ranks_data"),
        size,
        "pruned_ranks");

    ArraySlice<int32_t> output_pruned_indptr(output_pruned_indptr_array, "output_pruned_indptr");
    ArraySlice<int32_t> output_pruned_indices(output_pruned_indices_array, "output_pruned_indices");
    ArraySlice<float32_t> output_pruned_ranks(output_pruned_ranks_array, "output_pruned_ranks");

    FastAssertCompare(output_pruned_indptr.size(), ==, size + 1);
    FastAssertCompare(output_pruned_indices.size(), >=, size * pruned_degree);
    FastAssertCompare(output_pruned_ranks.size(), >=, size * pruned_degree);

    int start_position = output_pruned_indptr[0] = 0;
    for (int row_index = 0; row_index < size; ++row_index) {
        FastAssertCompare(start_position, ==, output_pruned_indptr[row_index]);
        auto input_ranks = input_pruned_ranks.get_row_data(row_index);
        if (input_ranks.size() <= pruned_degree) {
            start_position += input_ranks.size();
        } else {
            start_position += pruned_degree;
        }
        output_pruned_indptr[row_index + 1] = start_position;
    }

#pragma omp parallel for
    for (int row_index = 0; row_index < size; ++row_index) {
        prune_row(row_index,
                  pruned_degree,
                  input_pruned_ranks,
                  output_pruned_indptr,
                  output_pruned_indices,
                  output_pruned_ranks);
    }
}

}  // namespace metacells

PYBIND11_MODULE(extensions, module) {
    module.doc() = "C++ extensions to support the metacells package.";
#define REGISTER_D_O(D, O)                          \
    module.def("downsample_array_" #D "_" #O,       \
               &metacells::downsample_array<D, O>,  \
               "Downsample array.");                \
    module.def("downsample_matrix_" #D "_" #O,      \
               &metacells::downsample_matrix<D, O>, \
               "Downsample matrix.");

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

#define REGISTER_D_P_O(D, P, O)                            \
    module.def("downsample_compressed_" #D "_" #P "_" #O,  \
               &metacells::downsample_compressed<D, P, O>, \
               "Downsample compressed data.");

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
    REGISTER_D_P_O(float32_t, uint64_t, int32_t)
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
    REGISTER_D_P_O(float64_t, uint64_t, int32_t)
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
    REGISTER_D_P_O(int32_t, uint64_t, int32_t)
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
    REGISTER_D_P_O(int64_t, uint64_t, int32_t)
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
    REGISTER_D_P_O(uint32_t, uint64_t, int32_t)
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
    REGISTER_D_P_O(uint64_t, uint64_t, int32_t)
    REGISTER_D_P_O(uint64_t, uint64_t, int64_t)
    REGISTER_D_P_O(uint64_t, uint64_t, uint32_t)
    REGISTER_D_P_O(uint64_t, uint64_t, uint64_t)

#define REGISTER_D_I_P(D, I, P)                          \
    module.def("collect_compressed_" #D "_" #I "_" #P,   \
               &metacells::collect_compressed<D, I, P>,  \
               "Collect compressed data for relayout."); \
    module.def("sort_compressed_" #D "_" #I "_" #P,      \
               &metacells::sort_compressed<D, I, P>,     \
               "Sort compressed data.");

    REGISTER_D_I_P(float32_t, int32_t, int32_t)
    REGISTER_D_I_P(float32_t, int32_t, int64_t)
    REGISTER_D_I_P(float32_t, int32_t, uint32_t)
    REGISTER_D_I_P(float32_t, int32_t, uint64_t)
    REGISTER_D_I_P(float32_t, int64_t, int32_t)
    REGISTER_D_I_P(float32_t, int64_t, int64_t)
    REGISTER_D_I_P(float32_t, int64_t, uint32_t)
    REGISTER_D_I_P(float32_t, int64_t, uint64_t)
    REGISTER_D_I_P(float32_t, uint32_t, uint32_t)
    REGISTER_D_I_P(float32_t, uint32_t, uint64_t)
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
    REGISTER_D_I_P(float64_t, uint32_t, uint32_t)
    REGISTER_D_I_P(float64_t, uint32_t, uint64_t)
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
    REGISTER_D_I_P(int32_t, uint32_t, uint32_t)
    REGISTER_D_I_P(int32_t, uint32_t, uint64_t)
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
    REGISTER_D_I_P(int64_t, uint32_t, uint32_t)
    REGISTER_D_I_P(int64_t, uint32_t, uint64_t)
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
    REGISTER_D_I_P(uint32_t, uint32_t, uint32_t)
    REGISTER_D_I_P(uint32_t, uint32_t, uint64_t)
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
    REGISTER_D_I_P(uint64_t, uint32_t, uint32_t)
    REGISTER_D_I_P(uint64_t, uint32_t, uint64_t)
    REGISTER_D_I_P(uint64_t, uint64_t, uint32_t)
    REGISTER_D_I_P(uint64_t, uint64_t, uint64_t)

    module.def("collect_outgoing",
               &metacells::collect_outgoing,
               "Collect the topmost outgoing edges.");

    module.def("collect_pruned", &metacells::collect_pruned, "Collect the topmost pruned edges.");
}
