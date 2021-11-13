#include "metacells/extensions.h"

namespace metacells {

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

        SlowAssertCompare(tree[base_of_level + index_in_level] + tree[base_of_level + index_in_level + 1],
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

template<typename D, typename O>
static void
downsample_slice(ConstArraySlice<D> input, ArraySlice<O> output, const size_t samples, const size_t random_seed) {
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
downsample_dense(const pybind11::array_t<D>& input_matrix,
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

void
register_downsample(pybind11::module& module) {
#define REGISTER_D_O(D, O)                                                                        \
    module.def("downsample_array_" #D "_" #O, &downsample_array<D, O>, "Downsample array data."); \
    module.def("downsample_dense_" #D "_" #O, &downsample_dense<D, O>, "Downsample dense matrix data.");

#define REGISTER_DS_O(O)       \
    REGISTER_D_O(int8_t, O)    \
    REGISTER_D_O(int16_t, O)   \
    REGISTER_D_O(int32_t, O)   \
    REGISTER_D_O(int64_t, O)   \
    REGISTER_D_O(uint8_t, O)   \
    REGISTER_D_O(uint16_t, O)  \
    REGISTER_D_O(uint32_t, O)  \
    REGISTER_D_O(uint64_t, O)  \
    REGISTER_D_O(float32_t, O) \
    REGISTER_D_O(float64_t, O)

    REGISTER_DS_O(int8_t)
    REGISTER_DS_O(int16_t)
    REGISTER_DS_O(int32_t)
    REGISTER_DS_O(int64_t)
    REGISTER_DS_O(uint8_t)
    REGISTER_DS_O(uint16_t)
    REGISTER_DS_O(uint32_t)
    REGISTER_DS_O(uint64_t)
    REGISTER_DS_O(float32_t)
    REGISTER_DS_O(float64_t)

#define REGISTER_D_P_O(D, P, O)                           \
    module.def("downsample_compressed_" #D "_" #P "_" #O, \
               &downsample_compressed<D, P, O>,           \
               "Downsample compressed matrix data.");

#define REGISTER_DS_P_O(P, O)       \
    REGISTER_D_P_O(int8_t, P, O)    \
    REGISTER_D_P_O(int16_t, P, O)   \
    REGISTER_D_P_O(int32_t, P, O)   \
    REGISTER_D_P_O(int64_t, P, O)   \
    REGISTER_D_P_O(uint8_t, P, O)   \
    REGISTER_D_P_O(uint16_t, P, O)  \
    REGISTER_D_P_O(uint32_t, P, O)  \
    REGISTER_D_P_O(uint64_t, P, O)  \
    REGISTER_D_P_O(float32_t, P, O) \
    REGISTER_D_P_O(float64_t, P, O)

#define REGISTER_DS_PS_O(O)      \
    REGISTER_DS_P_O(int32_t, O)  \
    REGISTER_DS_P_O(int64_t, O)  \
    REGISTER_DS_P_O(uint32_t, O) \
    REGISTER_DS_P_O(uint64_t, O)

    REGISTER_DS_PS_O(int8_t)
    REGISTER_DS_PS_O(int16_t)
    REGISTER_DS_PS_O(int32_t)
    REGISTER_DS_PS_O(int64_t)
    REGISTER_DS_PS_O(uint8_t)
    REGISTER_DS_PS_O(uint16_t)
    REGISTER_DS_PS_O(uint32_t)
    REGISTER_DS_PS_O(uint64_t)
    REGISTER_DS_PS_O(float32_t)
    REGISTER_DS_PS_O(float64_t)
}

}
