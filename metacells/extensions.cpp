/// C++ extensions to support the metacells package.";

#if ASSERT_LEVEL == 0
#    define FastAssertCompare(...)
#    define FastAssertCompareWhat(...)
#    define SlowAssertCompare(...)
#    define SlowAssertCompareWhat(...)

#else  // ASSERT_LEVEL > 0
#    undef NDEBUG
#    include <iostream>

#    define FastAssertCompare(X, OP, Y)                                                          \
        if (!(double(X) OP double(Y))) {                                                         \
            std::cerr << __FILE__ << ":" << __LINE__ << ": failed assert: " << #X << " -> " << X \
                      << " " << #OP << " " << Y << " <- " << #Y << "" << std::endl;              \
            assert(false);                                                                       \
        } else
#    define FastAssertCompareWhat(X, OP, Y, WHAT)                                                  \
        if (!(double(X) OP double(Y))) {                                                           \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " << WHAT << ": failed assert: " << #X  \
                      << " -> " << X << " " << #OP << " " << Y << " <- " << #Y << "" << std::endl; \
            assert(false);                                                                         \
        } else

#    if ASSERT_LEVEL == 1
#        define SlowAssertCompare(...)
#        define SlowAssertCompareWhat(...)

#    elif ASSERT_LEVEL == 2
#        define SlowAssertCompare(...) FastAssertCompare(__VA_ARGS__)
#        define SlowAssertCompareWhat(...) FastAssertCompareWhat(__VA_ARGS__)

#    else
#        error Invalid ASSERT_LEVEL
#    endif
#endif

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <cmath>
#include <random>

namespace metacells {

template<typename T>
void
fake_use(T&) {}

static const double LOG2_SCALE = 1.0 / log(2.0);

double
log2(const double x) {
    FastAssertCompare(x, >, 0);
    return log(x) * LOG2_SCALE;
}

/// Range of indices for loops.
template<typename T>
class Range {
private:
    T m_size;
    T m_iter;

public:
    /// Construct a range of some size.
    Range(const T size) : m_size(size), m_iter(0) {}

    /// Fake start element for loops.
    const Range& begin() const { return *this; }

    /// Fake stop element for loops.
    const Range& end() const { return *this; }

    /// Comparison for loops.
    bool operator!=(const Range&) const { return m_iter < m_size; }

    /// Increment for loops.
    void operator++() { ++m_iter; }

    /// Access index value for loops.
    T operator*() const { return m_iter; }

    /// The number of indices in the range.
    T size() const { return m_size; }
};

/// An immutable contiguous slice of an array of type ``T``.
template<typename T>
class ConstArraySlice {
private:
    const T* m_data;     ///< Pointer to the first element.
    ssize_t m_size;      ///< Number of elements.
    const char* m_name;  ///< Name for error messages.

public:
    /// Construct an array slice from raw data.
    ConstArraySlice(const T* const data, const ssize_t size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    /// Construct an array slice based on a Python Numpy array.
    ConstArraySlice(const pybind11::array_t<T>& array, const char* const name)
      : m_data(array.data()), m_size(array.size()), m_name(name) {
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
        return ArraySlice(m_data + start, stop - start, m_name);
    }

    /// The number of elements in the slice.
    ssize_t size() const { return m_size; }

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
    ssize_t m_size;      ///< Number of elements.
    const char* m_name;  ///< Name for error messages.

public:
    /// Construct an array slice from raw data.
    ArraySlice(T* const data, const ssize_t size, const char* const name)
      : m_data(data), m_size(size), m_name(name) {}

    /// Construct an array slice based on a Python Numpy array.
    ArraySlice(pybind11::array_t<T>& array, const char* name)
      : m_data(array.mutable_data()), m_size(array.size()), m_name(name) {
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
    ssize_t size() const { return m_size; }

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

template<typename T>
static T
ceil_power_of_two(T size) {
    return 1 << int(ceil(log2(size)));
}

static int
downsample_tmp_size(int size) {
    if (size <= 1) {
        return 0;
    }
    return 2 * ceil_power_of_two(size) - 1;
}

template<typename D, typename T>
static void
initialize_tree(ConstArraySlice<D> input, ArraySlice<T> tree) {
    FastAssertCompare(input.size(), >=, 2);

    int input_size = ceil_power_of_two(int(input.size()));
    std::copy(input.begin(), input.end(), tree.begin());
    std::fill(tree.begin() + input.size(), tree.begin() + input_size, 0);

    while (input_size > 1) {
        auto slices = tree.split(input_size);
        auto input = slices.first;
        tree = slices.second;

        input_size /= 2;
        for (auto index : Range<int>(input_size)) {
            tree[index] = input[index * 2] + input[index * 2 + 1];
        }
    }

    FastAssertCompare(tree.size(), ==, 1);
}

template<typename T>
static int
random_sample(ArraySlice<T> tree, int64_t random) {
    int size_of_level = 1;
    int base_of_level = int(tree.size()) - 1;
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
        int64_t right_random = random - tree[index_in_tree];
        if (right_random >= 0) {
            ++index_in_level;
            ++index_in_tree;
            SlowAssertCompare(index_in_level, <, size_of_level);
            random = right_random;
        }
    }
}

/// See the Python metacell.utilities.computation.downsample
template<typename D, typename T, typename O>
static void
typed_downsample(const pybind11::array_t<D>& input_array,
                 pybind11::array_t<T>& tree_array,
                 pybind11::array_t<O>& output_array,
                 const int samples,
                 const int random_seed) {
    ConstArraySlice<D> input{ input_array, "input_array" };
    ArraySlice<T> tree{ tree_array, "tree_array" };
    ArraySlice<O> output{ output_array, "output_array" };
    std::minstd_rand random(random_seed);

    FastAssertCompare(samples, >=, 0);
    FastAssertCompare(tree.size(), ==, downsample_tmp_size(input.size()));
    FastAssertCompare(output.size(), ==, input.size());

    if (input.size() == 0) {
        return;
    }

    if (input.size() == 1) {
        output[0] = double(samples) < double(input[0]) ? samples : input[0];
        return;
    }

    initialize_tree(input, tree);
    T& total = tree[tree.size() - 1];

    if (double(total) <= double(samples)) {
        if (static_cast<const void*>(output.begin()) != static_cast<const void*>(input.begin())) {
            std::copy(input.begin(), input.end(), output.begin());
        }
        return;
    }

    std::fill(output.begin(), output.end(), 0);

    for (auto _ : Range<int>(samples)) {
        fake_use(_);
        ++output[random_sample(tree, random() % int64_t(total))];
    }
}

#define DEFINE_DOWNSAMPLE(D, T, O)                                                      \
    static void downsample_##D##_##T##_##O(const pybind11::array_t<D##_t>& input_array, \
                                           pybind11::array_t<T##_t>& tree_array,        \
                                           pybind11::array_t<O##_t>& output_array,      \
                                           const int samples,                           \
                                           const int random_seed) {                     \
        typed_downsample<D##_t, T##_t, O##_t>(input_array,                              \
                                              tree_array,                               \
                                              output_array,                             \
                                              samples,                                  \
                                              random_seed);                             \
    }

typedef float float32_t;
typedef double float64_t;

static_assert(sizeof(float32_t) == 4);
static_assert(sizeof(float64_t) == 8);

DEFINE_DOWNSAMPLE(float32, float32, float32)
DEFINE_DOWNSAMPLE(float32, float32, float64)
DEFINE_DOWNSAMPLE(float32, float32, int32)
DEFINE_DOWNSAMPLE(float32, float32, int64)
DEFINE_DOWNSAMPLE(float32, float32, uint32)
DEFINE_DOWNSAMPLE(float32, float32, uint64)
DEFINE_DOWNSAMPLE(float32, float64, float32)
DEFINE_DOWNSAMPLE(float32, float64, float64)
DEFINE_DOWNSAMPLE(float32, float64, int32)
DEFINE_DOWNSAMPLE(float32, float64, int64)
DEFINE_DOWNSAMPLE(float32, float64, uint32)
DEFINE_DOWNSAMPLE(float32, float64, uint64)
DEFINE_DOWNSAMPLE(float32, int32, float32)
DEFINE_DOWNSAMPLE(float32, int32, float64)
DEFINE_DOWNSAMPLE(float32, int32, int32)
DEFINE_DOWNSAMPLE(float32, int32, int64)
DEFINE_DOWNSAMPLE(float32, int32, uint32)
DEFINE_DOWNSAMPLE(float32, int32, uint64)
DEFINE_DOWNSAMPLE(float32, int64, float32)
DEFINE_DOWNSAMPLE(float32, int64, float64)
DEFINE_DOWNSAMPLE(float32, int64, int32)
DEFINE_DOWNSAMPLE(float32, int64, int64)
DEFINE_DOWNSAMPLE(float32, int64, uint32)
DEFINE_DOWNSAMPLE(float32, int64, uint64)
DEFINE_DOWNSAMPLE(float32, uint32, float32)
DEFINE_DOWNSAMPLE(float32, uint32, float64)
DEFINE_DOWNSAMPLE(float32, uint32, int32)
DEFINE_DOWNSAMPLE(float32, uint32, int64)
DEFINE_DOWNSAMPLE(float32, uint32, uint32)
DEFINE_DOWNSAMPLE(float32, uint32, uint64)
DEFINE_DOWNSAMPLE(float32, uint64, float32)
DEFINE_DOWNSAMPLE(float32, uint64, float64)
DEFINE_DOWNSAMPLE(float32, uint64, int32)
DEFINE_DOWNSAMPLE(float32, uint64, int64)
DEFINE_DOWNSAMPLE(float32, uint64, uint32)
DEFINE_DOWNSAMPLE(float32, uint64, uint64)
DEFINE_DOWNSAMPLE(float64, float32, float32)
DEFINE_DOWNSAMPLE(float64, float32, float64)
DEFINE_DOWNSAMPLE(float64, float32, int32)
DEFINE_DOWNSAMPLE(float64, float32, int64)
DEFINE_DOWNSAMPLE(float64, float32, uint32)
DEFINE_DOWNSAMPLE(float64, float32, uint64)
DEFINE_DOWNSAMPLE(float64, float64, float32)
DEFINE_DOWNSAMPLE(float64, float64, float64)
DEFINE_DOWNSAMPLE(float64, float64, int32)
DEFINE_DOWNSAMPLE(float64, float64, int64)
DEFINE_DOWNSAMPLE(float64, float64, uint32)
DEFINE_DOWNSAMPLE(float64, float64, uint64)
DEFINE_DOWNSAMPLE(float64, int32, float32)
DEFINE_DOWNSAMPLE(float64, int32, float64)
DEFINE_DOWNSAMPLE(float64, int32, int32)
DEFINE_DOWNSAMPLE(float64, int32, int64)
DEFINE_DOWNSAMPLE(float64, int32, uint32)
DEFINE_DOWNSAMPLE(float64, int32, uint64)
DEFINE_DOWNSAMPLE(float64, int64, float32)
DEFINE_DOWNSAMPLE(float64, int64, float64)
DEFINE_DOWNSAMPLE(float64, int64, int32)
DEFINE_DOWNSAMPLE(float64, int64, int64)
DEFINE_DOWNSAMPLE(float64, int64, uint32)
DEFINE_DOWNSAMPLE(float64, int64, uint64)
DEFINE_DOWNSAMPLE(float64, uint32, float32)
DEFINE_DOWNSAMPLE(float64, uint32, float64)
DEFINE_DOWNSAMPLE(float64, uint32, int32)
DEFINE_DOWNSAMPLE(float64, uint32, int64)
DEFINE_DOWNSAMPLE(float64, uint32, uint32)
DEFINE_DOWNSAMPLE(float64, uint32, uint64)
DEFINE_DOWNSAMPLE(float64, uint64, float32)
DEFINE_DOWNSAMPLE(float64, uint64, float64)
DEFINE_DOWNSAMPLE(float64, uint64, int32)
DEFINE_DOWNSAMPLE(float64, uint64, int64)
DEFINE_DOWNSAMPLE(float64, uint64, uint32)
DEFINE_DOWNSAMPLE(float64, uint64, uint64)
DEFINE_DOWNSAMPLE(int32, float32, float32)
DEFINE_DOWNSAMPLE(int32, float32, float64)
DEFINE_DOWNSAMPLE(int32, float32, int32)
DEFINE_DOWNSAMPLE(int32, float32, int64)
DEFINE_DOWNSAMPLE(int32, float32, uint32)
DEFINE_DOWNSAMPLE(int32, float32, uint64)
DEFINE_DOWNSAMPLE(int32, float64, float32)
DEFINE_DOWNSAMPLE(int32, float64, float64)
DEFINE_DOWNSAMPLE(int32, float64, int32)
DEFINE_DOWNSAMPLE(int32, float64, int64)
DEFINE_DOWNSAMPLE(int32, float64, uint32)
DEFINE_DOWNSAMPLE(int32, float64, uint64)
DEFINE_DOWNSAMPLE(int32, int32, float32)
DEFINE_DOWNSAMPLE(int32, int32, float64)
DEFINE_DOWNSAMPLE(int32, int32, int32)
DEFINE_DOWNSAMPLE(int32, int32, int64)
DEFINE_DOWNSAMPLE(int32, int32, uint32)
DEFINE_DOWNSAMPLE(int32, int32, uint64)
DEFINE_DOWNSAMPLE(int32, int64, float32)
DEFINE_DOWNSAMPLE(int32, int64, float64)
DEFINE_DOWNSAMPLE(int32, int64, int32)
DEFINE_DOWNSAMPLE(int32, int64, int64)
DEFINE_DOWNSAMPLE(int32, int64, uint32)
DEFINE_DOWNSAMPLE(int32, int64, uint64)
DEFINE_DOWNSAMPLE(int32, uint32, float32)
DEFINE_DOWNSAMPLE(int32, uint32, float64)
DEFINE_DOWNSAMPLE(int32, uint32, int32)
DEFINE_DOWNSAMPLE(int32, uint32, int64)
DEFINE_DOWNSAMPLE(int32, uint32, uint32)
DEFINE_DOWNSAMPLE(int32, uint32, uint64)
DEFINE_DOWNSAMPLE(int32, uint64, float32)
DEFINE_DOWNSAMPLE(int32, uint64, float64)
DEFINE_DOWNSAMPLE(int32, uint64, int32)
DEFINE_DOWNSAMPLE(int32, uint64, int64)
DEFINE_DOWNSAMPLE(int32, uint64, uint32)
DEFINE_DOWNSAMPLE(int32, uint64, uint64)
DEFINE_DOWNSAMPLE(int64, float32, float32)
DEFINE_DOWNSAMPLE(int64, float32, float64)
DEFINE_DOWNSAMPLE(int64, float32, int32)
DEFINE_DOWNSAMPLE(int64, float32, int64)
DEFINE_DOWNSAMPLE(int64, float32, uint32)
DEFINE_DOWNSAMPLE(int64, float32, uint64)
DEFINE_DOWNSAMPLE(int64, float64, float32)
DEFINE_DOWNSAMPLE(int64, float64, float64)
DEFINE_DOWNSAMPLE(int64, float64, int32)
DEFINE_DOWNSAMPLE(int64, float64, int64)
DEFINE_DOWNSAMPLE(int64, float64, uint32)
DEFINE_DOWNSAMPLE(int64, float64, uint64)
DEFINE_DOWNSAMPLE(int64, int32, float32)
DEFINE_DOWNSAMPLE(int64, int32, float64)
DEFINE_DOWNSAMPLE(int64, int32, int32)
DEFINE_DOWNSAMPLE(int64, int32, int64)
DEFINE_DOWNSAMPLE(int64, int32, uint32)
DEFINE_DOWNSAMPLE(int64, int32, uint64)
DEFINE_DOWNSAMPLE(int64, int64, float32)
DEFINE_DOWNSAMPLE(int64, int64, float64)
DEFINE_DOWNSAMPLE(int64, int64, int32)
DEFINE_DOWNSAMPLE(int64, int64, int64)
DEFINE_DOWNSAMPLE(int64, int64, uint32)
DEFINE_DOWNSAMPLE(int64, int64, uint64)
DEFINE_DOWNSAMPLE(int64, uint32, float32)
DEFINE_DOWNSAMPLE(int64, uint32, float64)
DEFINE_DOWNSAMPLE(int64, uint32, int32)
DEFINE_DOWNSAMPLE(int64, uint32, int64)
DEFINE_DOWNSAMPLE(int64, uint32, uint32)
DEFINE_DOWNSAMPLE(int64, uint32, uint64)
DEFINE_DOWNSAMPLE(int64, uint64, float32)
DEFINE_DOWNSAMPLE(int64, uint64, float64)
DEFINE_DOWNSAMPLE(int64, uint64, int32)
DEFINE_DOWNSAMPLE(int64, uint64, int64)
DEFINE_DOWNSAMPLE(int64, uint64, uint32)
DEFINE_DOWNSAMPLE(int64, uint64, uint64)
DEFINE_DOWNSAMPLE(uint32, float32, float32)
DEFINE_DOWNSAMPLE(uint32, float32, float64)
DEFINE_DOWNSAMPLE(uint32, float32, int32)
DEFINE_DOWNSAMPLE(uint32, float32, int64)
DEFINE_DOWNSAMPLE(uint32, float32, uint32)
DEFINE_DOWNSAMPLE(uint32, float32, uint64)
DEFINE_DOWNSAMPLE(uint32, float64, float32)
DEFINE_DOWNSAMPLE(uint32, float64, float64)
DEFINE_DOWNSAMPLE(uint32, float64, int32)
DEFINE_DOWNSAMPLE(uint32, float64, int64)
DEFINE_DOWNSAMPLE(uint32, float64, uint32)
DEFINE_DOWNSAMPLE(uint32, float64, uint64)
DEFINE_DOWNSAMPLE(uint32, int32, float32)
DEFINE_DOWNSAMPLE(uint32, int32, float64)
DEFINE_DOWNSAMPLE(uint32, int32, int32)
DEFINE_DOWNSAMPLE(uint32, int32, int64)
DEFINE_DOWNSAMPLE(uint32, int32, uint32)
DEFINE_DOWNSAMPLE(uint32, int32, uint64)
DEFINE_DOWNSAMPLE(uint32, int64, float32)
DEFINE_DOWNSAMPLE(uint32, int64, float64)
DEFINE_DOWNSAMPLE(uint32, int64, int32)
DEFINE_DOWNSAMPLE(uint32, int64, int64)
DEFINE_DOWNSAMPLE(uint32, int64, uint32)
DEFINE_DOWNSAMPLE(uint32, int64, uint64)
DEFINE_DOWNSAMPLE(uint32, uint32, float32)
DEFINE_DOWNSAMPLE(uint32, uint32, float64)
DEFINE_DOWNSAMPLE(uint32, uint32, int32)
DEFINE_DOWNSAMPLE(uint32, uint32, int64)
DEFINE_DOWNSAMPLE(uint32, uint32, uint32)
DEFINE_DOWNSAMPLE(uint32, uint32, uint64)
DEFINE_DOWNSAMPLE(uint32, uint64, float32)
DEFINE_DOWNSAMPLE(uint32, uint64, float64)
DEFINE_DOWNSAMPLE(uint32, uint64, int32)
DEFINE_DOWNSAMPLE(uint32, uint64, int64)
DEFINE_DOWNSAMPLE(uint32, uint64, uint32)
DEFINE_DOWNSAMPLE(uint32, uint64, uint64)
DEFINE_DOWNSAMPLE(uint64, float32, float32)
DEFINE_DOWNSAMPLE(uint64, float32, float64)
DEFINE_DOWNSAMPLE(uint64, float32, int32)
DEFINE_DOWNSAMPLE(uint64, float32, int64)
DEFINE_DOWNSAMPLE(uint64, float32, uint32)
DEFINE_DOWNSAMPLE(uint64, float32, uint64)
DEFINE_DOWNSAMPLE(uint64, float64, float32)
DEFINE_DOWNSAMPLE(uint64, float64, float64)
DEFINE_DOWNSAMPLE(uint64, float64, int32)
DEFINE_DOWNSAMPLE(uint64, float64, int64)
DEFINE_DOWNSAMPLE(uint64, float64, uint32)
DEFINE_DOWNSAMPLE(uint64, float64, uint64)
DEFINE_DOWNSAMPLE(uint64, int32, float32)
DEFINE_DOWNSAMPLE(uint64, int32, float64)
DEFINE_DOWNSAMPLE(uint64, int32, int32)
DEFINE_DOWNSAMPLE(uint64, int32, int64)
DEFINE_DOWNSAMPLE(uint64, int32, uint32)
DEFINE_DOWNSAMPLE(uint64, int32, uint64)
DEFINE_DOWNSAMPLE(uint64, int64, float32)
DEFINE_DOWNSAMPLE(uint64, int64, float64)
DEFINE_DOWNSAMPLE(uint64, int64, int32)
DEFINE_DOWNSAMPLE(uint64, int64, int64)
DEFINE_DOWNSAMPLE(uint64, int64, uint32)
DEFINE_DOWNSAMPLE(uint64, int64, uint64)
DEFINE_DOWNSAMPLE(uint64, uint32, float32)
DEFINE_DOWNSAMPLE(uint64, uint32, float64)
DEFINE_DOWNSAMPLE(uint64, uint32, int32)
DEFINE_DOWNSAMPLE(uint64, uint32, int64)
DEFINE_DOWNSAMPLE(uint64, uint32, uint32)
DEFINE_DOWNSAMPLE(uint64, uint32, uint64)
DEFINE_DOWNSAMPLE(uint64, uint64, float32)
DEFINE_DOWNSAMPLE(uint64, uint64, float64)
DEFINE_DOWNSAMPLE(uint64, uint64, int32)
DEFINE_DOWNSAMPLE(uint64, uint64, int64)
DEFINE_DOWNSAMPLE(uint64, uint64, uint32)
DEFINE_DOWNSAMPLE(uint64, uint64, uint64)

}  // namespace metacells

PYBIND11_MODULE(extensions, module) {
    module.doc() = "C++ extensions to support the metacells package.";
    module.def("downsample_tmp_size",
               &metacells::downsample_tmp_size,
               "Size needed for downsample temporary array.");
#define REGISTER_DOWNSAMPLE(D, T, O)                   \
    module.def("downsample_" #D "_" #T "_" #O,         \
               &metacells::downsample_##D##_##T##_##O, \
               "Downsample array of sample counts.")

    REGISTER_DOWNSAMPLE(float32, float32, float32);
    REGISTER_DOWNSAMPLE(float32, float32, float64);
    REGISTER_DOWNSAMPLE(float32, float32, int32);
    REGISTER_DOWNSAMPLE(float32, float32, int64);
    REGISTER_DOWNSAMPLE(float32, float32, uint32);
    REGISTER_DOWNSAMPLE(float32, float32, uint64);
    REGISTER_DOWNSAMPLE(float32, float64, float32);
    REGISTER_DOWNSAMPLE(float32, float64, float64);
    REGISTER_DOWNSAMPLE(float32, float64, int32);
    REGISTER_DOWNSAMPLE(float32, float64, int64);
    REGISTER_DOWNSAMPLE(float32, float64, uint32);
    REGISTER_DOWNSAMPLE(float32, float64, uint64);
    REGISTER_DOWNSAMPLE(float32, int32, float32);
    REGISTER_DOWNSAMPLE(float32, int32, float64);
    REGISTER_DOWNSAMPLE(float32, int32, int32);
    REGISTER_DOWNSAMPLE(float32, int32, int64);
    REGISTER_DOWNSAMPLE(float32, int32, uint32);
    REGISTER_DOWNSAMPLE(float32, int32, uint64);
    REGISTER_DOWNSAMPLE(float32, int64, float32);
    REGISTER_DOWNSAMPLE(float32, int64, float64);
    REGISTER_DOWNSAMPLE(float32, int64, int32);
    REGISTER_DOWNSAMPLE(float32, int64, int64);
    REGISTER_DOWNSAMPLE(float32, int64, uint32);
    REGISTER_DOWNSAMPLE(float32, int64, uint64);
    REGISTER_DOWNSAMPLE(float32, uint32, float32);
    REGISTER_DOWNSAMPLE(float32, uint32, float64);
    REGISTER_DOWNSAMPLE(float32, uint32, int32);
    REGISTER_DOWNSAMPLE(float32, uint32, int64);
    REGISTER_DOWNSAMPLE(float32, uint32, uint32);
    REGISTER_DOWNSAMPLE(float32, uint32, uint64);
    REGISTER_DOWNSAMPLE(float32, uint64, float32);
    REGISTER_DOWNSAMPLE(float32, uint64, float64);
    REGISTER_DOWNSAMPLE(float32, uint64, int32);
    REGISTER_DOWNSAMPLE(float32, uint64, int64);
    REGISTER_DOWNSAMPLE(float32, uint64, uint32);
    REGISTER_DOWNSAMPLE(float32, uint64, uint64);
    REGISTER_DOWNSAMPLE(float64, float32, float32);
    REGISTER_DOWNSAMPLE(float64, float32, float64);
    REGISTER_DOWNSAMPLE(float64, float32, int32);
    REGISTER_DOWNSAMPLE(float64, float32, int64);
    REGISTER_DOWNSAMPLE(float64, float32, uint32);
    REGISTER_DOWNSAMPLE(float64, float32, uint64);
    REGISTER_DOWNSAMPLE(float64, float64, float32);
    REGISTER_DOWNSAMPLE(float64, float64, float64);
    REGISTER_DOWNSAMPLE(float64, float64, int32);
    REGISTER_DOWNSAMPLE(float64, float64, int64);
    REGISTER_DOWNSAMPLE(float64, float64, uint32);
    REGISTER_DOWNSAMPLE(float64, float64, uint64);
    REGISTER_DOWNSAMPLE(float64, int32, float32);
    REGISTER_DOWNSAMPLE(float64, int32, float64);
    REGISTER_DOWNSAMPLE(float64, int32, int32);
    REGISTER_DOWNSAMPLE(float64, int32, int64);
    REGISTER_DOWNSAMPLE(float64, int32, uint32);
    REGISTER_DOWNSAMPLE(float64, int32, uint64);
    REGISTER_DOWNSAMPLE(float64, int64, float32);
    REGISTER_DOWNSAMPLE(float64, int64, float64);
    REGISTER_DOWNSAMPLE(float64, int64, int32);
    REGISTER_DOWNSAMPLE(float64, int64, int64);
    REGISTER_DOWNSAMPLE(float64, int64, uint32);
    REGISTER_DOWNSAMPLE(float64, int64, uint64);
    REGISTER_DOWNSAMPLE(float64, uint32, float32);
    REGISTER_DOWNSAMPLE(float64, uint32, float64);
    REGISTER_DOWNSAMPLE(float64, uint32, int32);
    REGISTER_DOWNSAMPLE(float64, uint32, int64);
    REGISTER_DOWNSAMPLE(float64, uint32, uint32);
    REGISTER_DOWNSAMPLE(float64, uint32, uint64);
    REGISTER_DOWNSAMPLE(float64, uint64, float32);
    REGISTER_DOWNSAMPLE(float64, uint64, float64);
    REGISTER_DOWNSAMPLE(float64, uint64, int32);
    REGISTER_DOWNSAMPLE(float64, uint64, int64);
    REGISTER_DOWNSAMPLE(float64, uint64, uint32);
    REGISTER_DOWNSAMPLE(float64, uint64, uint64);
    REGISTER_DOWNSAMPLE(int32, float32, float32);
    REGISTER_DOWNSAMPLE(int32, float32, float64);
    REGISTER_DOWNSAMPLE(int32, float32, int32);
    REGISTER_DOWNSAMPLE(int32, float32, int64);
    REGISTER_DOWNSAMPLE(int32, float32, uint32);
    REGISTER_DOWNSAMPLE(int32, float32, uint64);
    REGISTER_DOWNSAMPLE(int32, float64, float32);
    REGISTER_DOWNSAMPLE(int32, float64, float64);
    REGISTER_DOWNSAMPLE(int32, float64, int32);
    REGISTER_DOWNSAMPLE(int32, float64, int64);
    REGISTER_DOWNSAMPLE(int32, float64, uint32);
    REGISTER_DOWNSAMPLE(int32, float64, uint64);
    REGISTER_DOWNSAMPLE(int32, int32, float32);
    REGISTER_DOWNSAMPLE(int32, int32, float64);
    REGISTER_DOWNSAMPLE(int32, int32, int32);
    REGISTER_DOWNSAMPLE(int32, int32, int64);
    REGISTER_DOWNSAMPLE(int32, int32, uint32);
    REGISTER_DOWNSAMPLE(int32, int32, uint64);
    REGISTER_DOWNSAMPLE(int32, int64, float32);
    REGISTER_DOWNSAMPLE(int32, int64, float64);
    REGISTER_DOWNSAMPLE(int32, int64, int32);
    REGISTER_DOWNSAMPLE(int32, int64, int64);
    REGISTER_DOWNSAMPLE(int32, int64, uint32);
    REGISTER_DOWNSAMPLE(int32, int64, uint64);
    REGISTER_DOWNSAMPLE(int32, uint32, float32);
    REGISTER_DOWNSAMPLE(int32, uint32, float64);
    REGISTER_DOWNSAMPLE(int32, uint32, int32);
    REGISTER_DOWNSAMPLE(int32, uint32, int64);
    REGISTER_DOWNSAMPLE(int32, uint32, uint32);
    REGISTER_DOWNSAMPLE(int32, uint32, uint64);
    REGISTER_DOWNSAMPLE(int32, uint64, float32);
    REGISTER_DOWNSAMPLE(int32, uint64, float64);
    REGISTER_DOWNSAMPLE(int32, uint64, int32);
    REGISTER_DOWNSAMPLE(int32, uint64, int64);
    REGISTER_DOWNSAMPLE(int32, uint64, uint32);
    REGISTER_DOWNSAMPLE(int32, uint64, uint64);
    REGISTER_DOWNSAMPLE(int64, float32, float32);
    REGISTER_DOWNSAMPLE(int64, float32, float64);
    REGISTER_DOWNSAMPLE(int64, float32, int32);
    REGISTER_DOWNSAMPLE(int64, float32, int64);
    REGISTER_DOWNSAMPLE(int64, float32, uint32);
    REGISTER_DOWNSAMPLE(int64, float32, uint64);
    REGISTER_DOWNSAMPLE(int64, float64, float32);
    REGISTER_DOWNSAMPLE(int64, float64, float64);
    REGISTER_DOWNSAMPLE(int64, float64, int32);
    REGISTER_DOWNSAMPLE(int64, float64, int64);
    REGISTER_DOWNSAMPLE(int64, float64, uint32);
    REGISTER_DOWNSAMPLE(int64, float64, uint64);
    REGISTER_DOWNSAMPLE(int64, int32, float32);
    REGISTER_DOWNSAMPLE(int64, int32, float64);
    REGISTER_DOWNSAMPLE(int64, int32, int32);
    REGISTER_DOWNSAMPLE(int64, int32, int64);
    REGISTER_DOWNSAMPLE(int64, int32, uint32);
    REGISTER_DOWNSAMPLE(int64, int32, uint64);
    REGISTER_DOWNSAMPLE(int64, int64, float32);
    REGISTER_DOWNSAMPLE(int64, int64, float64);
    REGISTER_DOWNSAMPLE(int64, int64, int32);
    REGISTER_DOWNSAMPLE(int64, int64, int64);
    REGISTER_DOWNSAMPLE(int64, int64, uint32);
    REGISTER_DOWNSAMPLE(int64, int64, uint64);
    REGISTER_DOWNSAMPLE(int64, uint32, float32);
    REGISTER_DOWNSAMPLE(int64, uint32, float64);
    REGISTER_DOWNSAMPLE(int64, uint32, int32);
    REGISTER_DOWNSAMPLE(int64, uint32, int64);
    REGISTER_DOWNSAMPLE(int64, uint32, uint32);
    REGISTER_DOWNSAMPLE(int64, uint32, uint64);
    REGISTER_DOWNSAMPLE(int64, uint64, float32);
    REGISTER_DOWNSAMPLE(int64, uint64, float64);
    REGISTER_DOWNSAMPLE(int64, uint64, int32);
    REGISTER_DOWNSAMPLE(int64, uint64, int64);
    REGISTER_DOWNSAMPLE(int64, uint64, uint32);
    REGISTER_DOWNSAMPLE(int64, uint64, uint64);
    REGISTER_DOWNSAMPLE(uint32, float32, float32);
    REGISTER_DOWNSAMPLE(uint32, float32, float64);
    REGISTER_DOWNSAMPLE(uint32, float32, int32);
    REGISTER_DOWNSAMPLE(uint32, float32, int64);
    REGISTER_DOWNSAMPLE(uint32, float32, uint32);
    REGISTER_DOWNSAMPLE(uint32, float32, uint64);
    REGISTER_DOWNSAMPLE(uint32, float64, float32);
    REGISTER_DOWNSAMPLE(uint32, float64, float64);
    REGISTER_DOWNSAMPLE(uint32, float64, int32);
    REGISTER_DOWNSAMPLE(uint32, float64, int64);
    REGISTER_DOWNSAMPLE(uint32, float64, uint32);
    REGISTER_DOWNSAMPLE(uint32, float64, uint64);
    REGISTER_DOWNSAMPLE(uint32, int32, float32);
    REGISTER_DOWNSAMPLE(uint32, int32, float64);
    REGISTER_DOWNSAMPLE(uint32, int32, int32);
    REGISTER_DOWNSAMPLE(uint32, int32, int64);
    REGISTER_DOWNSAMPLE(uint32, int32, uint32);
    REGISTER_DOWNSAMPLE(uint32, int32, uint64);
    REGISTER_DOWNSAMPLE(uint32, int64, float32);
    REGISTER_DOWNSAMPLE(uint32, int64, float64);
    REGISTER_DOWNSAMPLE(uint32, int64, int32);
    REGISTER_DOWNSAMPLE(uint32, int64, int64);
    REGISTER_DOWNSAMPLE(uint32, int64, uint32);
    REGISTER_DOWNSAMPLE(uint32, int64, uint64);
    REGISTER_DOWNSAMPLE(uint32, uint32, float32);
    REGISTER_DOWNSAMPLE(uint32, uint32, float64);
    REGISTER_DOWNSAMPLE(uint32, uint32, int32);
    REGISTER_DOWNSAMPLE(uint32, uint32, int64);
    REGISTER_DOWNSAMPLE(uint32, uint32, uint32);
    REGISTER_DOWNSAMPLE(uint32, uint32, uint64);
    REGISTER_DOWNSAMPLE(uint32, uint64, float32);
    REGISTER_DOWNSAMPLE(uint32, uint64, float64);
    REGISTER_DOWNSAMPLE(uint32, uint64, int32);
    REGISTER_DOWNSAMPLE(uint32, uint64, int64);
    REGISTER_DOWNSAMPLE(uint32, uint64, uint32);
    REGISTER_DOWNSAMPLE(uint32, uint64, uint64);
    REGISTER_DOWNSAMPLE(uint64, float32, float32);
    REGISTER_DOWNSAMPLE(uint64, float32, float64);
    REGISTER_DOWNSAMPLE(uint64, float32, int32);
    REGISTER_DOWNSAMPLE(uint64, float32, int64);
    REGISTER_DOWNSAMPLE(uint64, float32, uint32);
    REGISTER_DOWNSAMPLE(uint64, float32, uint64);
    REGISTER_DOWNSAMPLE(uint64, float64, float32);
    REGISTER_DOWNSAMPLE(uint64, float64, float64);
    REGISTER_DOWNSAMPLE(uint64, float64, int32);
    REGISTER_DOWNSAMPLE(uint64, float64, int64);
    REGISTER_DOWNSAMPLE(uint64, float64, uint32);
    REGISTER_DOWNSAMPLE(uint64, float64, uint64);
    REGISTER_DOWNSAMPLE(uint64, int32, float32);
    REGISTER_DOWNSAMPLE(uint64, int32, float64);
    REGISTER_DOWNSAMPLE(uint64, int32, int32);
    REGISTER_DOWNSAMPLE(uint64, int32, int64);
    REGISTER_DOWNSAMPLE(uint64, int32, uint32);
    REGISTER_DOWNSAMPLE(uint64, int32, uint64);
    REGISTER_DOWNSAMPLE(uint64, int64, float32);
    REGISTER_DOWNSAMPLE(uint64, int64, float64);
    REGISTER_DOWNSAMPLE(uint64, int64, int32);
    REGISTER_DOWNSAMPLE(uint64, int64, int64);
    REGISTER_DOWNSAMPLE(uint64, int64, uint32);
    REGISTER_DOWNSAMPLE(uint64, int64, uint64);
    REGISTER_DOWNSAMPLE(uint64, uint32, float32);
    REGISTER_DOWNSAMPLE(uint64, uint32, float64);
    REGISTER_DOWNSAMPLE(uint64, uint32, int32);
    REGISTER_DOWNSAMPLE(uint64, uint32, int64);
    REGISTER_DOWNSAMPLE(uint64, uint32, uint32);
    REGISTER_DOWNSAMPLE(uint64, uint32, uint64);
    REGISTER_DOWNSAMPLE(uint64, uint64, float32);
    REGISTER_DOWNSAMPLE(uint64, uint64, float64);
    REGISTER_DOWNSAMPLE(uint64, uint64, int32);
    REGISTER_DOWNSAMPLE(uint64, uint64, int64);
    REGISTER_DOWNSAMPLE(uint64, uint64, uint32);
    REGISTER_DOWNSAMPLE(uint64, uint64, uint64);
}
