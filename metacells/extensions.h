/// C++ extensions to support the metacells package.";

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace metacells {

#if ASSERT_LEVEL > 0
#    undef NDEBUG

extern std::mutex writer_mutex;

class AtomicWriter {
    std::ostringstream m_st;
    std::ostream& m_stream;

public:
    AtomicWriter(std::ostream& s = std::cerr) : m_stream(s) { m_st << std::this_thread::get_id() << ' '; }

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

extern thread_local AtomicWriter writer;

#elif ASSERT_LEVEL < 0 || ASSERT_LEVEL > 2
#    error Invalid ASSERT_LEVEL
#endif

#define LOCATED_LOG(COND) \
    if (!(COND))          \
        ;                 \
    else                  \
        writer << __FILE__ << ':' << __LINE__ << ':' << __FUNCTION__ << ":"

#if ASSERT_LEVEL >= 1
#    define FastAssertCompare(X, OP, Y)                                                                          \
        if (!(double(X) OP double(Y))) {                                                                         \
            std::lock_guard<std::mutex> io_lock(io_mutex);                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << ": failed assert: " << #X << " -> " << (X) << " " << #OP \
                      << " " << (Y) << " <- " << #Y << "" << std::endl;                                          \
            assert(false);                                                                                       \
        } else
#    define FastAssertCompareWhat(X, OP, Y, WHAT)                                                                  \
        if (!(double(X) OP double(Y))) {                                                                           \
            std::lock_guard<std::mutex> io_lock(io_mutex);                                                         \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " << WHAT << ": failed assert: " << #X << " -> " << (X) \
                      << " " << #OP << " " << (Y) << " <- " << #Y << "" << std::endl;                              \
            assert(false);                                                                                         \
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

typedef float float32_t;
typedef double float64_t;
typedef unsigned char uint8_t;
typedef unsigned int uint_t;

/*
#ifdef USE_AVX2
static std::ostream&
operator<<(std::ostream& os, const __m256d& avx2) {
    return os << "[" << ((float64_t*)&avx2)[0] << "," << ((float64_t*)&avx2)[1] << ","
              << ((float64_t*)&avx2)[2] << "," << ((float64_t*)&avx2)[3] << "]";
}

static std::ostream&
operator<<(std::ostream& os, const __m256& avx2) {
    return os << "[" << ((float32_t*)&avx2)[0] << "," << ((float32_t*)&avx2)[1] << ","
              << ((float32_t*)&avx2)[2] << "," << ((float32_t*)&avx2)[3] << ","
              << ((float32_t*)&avx2)[4] << "," << ((float32_t*)&avx2)[5] << ","
              << ((float32_t*)&avx2)[6] << "," << ((float32_t*)&avx2)[7] << "]";
}
#endif
*/

#if ASSERT_LEVEL > 0
extern std::mutex io_mutex;
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

static float64_t inline log2(const float64_t x) {
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

    ArraySlice(T* const data, const size_t size, const char* const name) : m_data(data), m_size(size), m_name(name) {}

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
    if (array.shape(0) == 0) {
        return 0;
    } else {
        return array.data(1, 0) - array.data(0, 0);
    }
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
      : ConstMatrixSlice(array.data(), array.shape(0), array.shape(1), matrix_step(array, name), name) {
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
      : MatrixSlice(array.mutable_data(), array.shape(0), array.shape(1), matrix_step(array, name), name) {
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

static const int size_t_count = 8;
extern thread_local bool g_size_t_used[size_t_count];
extern thread_local std::vector<size_t> g_size_t_vectors[size_t_count];

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
extern thread_local bool g_float64_t_used[float64_t_count];
extern thread_local std::vector<float64_t> g_float64_t_vectors[float64_t_count];

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

extern void
parallel_loop(const size_t size, std::function<void(size_t)> parallel_body, std::function<void(size_t)> serial_body);

static void inline parallel_loop(const size_t size, std::function<void(size_t)> parallel_body) {
    parallel_loop(size, parallel_body, parallel_body);
}

/*
static void inline
serial_loop(const size_t size, std::function<void(size_t)> serial_body) {
    for (size_t index = 0; index < size; ++index) {
        serial_body(index);
    }
}
*/

extern void
register_auroc(pybind11::module& module);
extern void
register_choose_seeds(pybind11::module& module);
extern void
register_correlate(pybind11::module& module);
extern void
register_cover(pybind11::module& module);
extern void
register_downsample(pybind11::module& module);
extern void
register_folds(pybind11::module& module);
extern void
register_logistics(pybind11::module& module);
extern void
register_partitions(pybind11::module& module);
extern void
register_prune_per(pybind11::module& module);
extern void
register_rank(pybind11::module& module);
extern void
register_relayout(pybind11::module& module);
extern void
register_shuffle(pybind11::module& module);
extern void
register_top_per(pybind11::module& module);

}  // namespace metacells
