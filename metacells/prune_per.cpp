#include "metacells/extensions.h"

namespace metacells {

template<typename D, typename I, typename P>
static void
prune_band(const size_t band_index,
           const size_t pruned_degree,
           ConstCompressedMatrix<D, I, P>& input_pruned_values,
           ArraySlice<D> output_pruned_values,
           ArraySlice<I> output_pruned_indices,
           ConstArraySlice<P> output_pruned_indptr) {
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
template<typename D, typename I, typename P>
static void
collect_pruned(const size_t pruned_degree,
               const pybind11::array_t<D>& input_pruned_values_data,
               const pybind11::array_t<I>& input_pruned_values_indices,
               const pybind11::array_t<P>& input_pruned_values_indptr,
               pybind11::array_t<D>& output_pruned_values_array,
               pybind11::array_t<I>& output_pruned_indices_array,
               pybind11::array_t<P>& output_pruned_indptr_array) {
    WithoutGil without_gil{};

    size_t size = input_pruned_values_indptr.size() - 1;
    ConstCompressedMatrix<D, I, P> input_pruned_values(ConstArraySlice<D>(input_pruned_values_data,
                                                                          "input_pruned_values_data"),
                                                       ConstArraySlice<I>(input_pruned_values_indices,
                                                                          "input_pruned_values_indices"),
                                                       ConstArraySlice<P>(input_pruned_values_indptr,
                                                                          "pruned_values_indptr"),
                                                       I(size),
                                                       "pruned_values");

    ArraySlice<D> output_pruned_values(output_pruned_values_array, "output_pruned_values");
    ArraySlice<I> output_pruned_indices(output_pruned_indices_array, "output_pruned_indices");
    ArraySlice<P> output_pruned_indptr(output_pruned_indptr_array, "output_pruned_indptr");

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
                   ConstArraySlice<P>(output_pruned_indptr));
    });
}

void
register_prune_per(pybind11::module& module) {
#define REGISTER_D_I_P(D, I, P) \
    module.def("collect_pruned_" #D "_" #I "_" #P, &collect_pruned<D, I, P>, "Collect the topmost pruned edges.");

#define REGISTER_DS_I_P(I, P)       \
    REGISTER_D_I_P(int8_t, I, P)    \
    REGISTER_D_I_P(int16_t, I, P)   \
    REGISTER_D_I_P(int32_t, I, P)   \
    REGISTER_D_I_P(int64_t, I, P)   \
    REGISTER_D_I_P(uint8_t, I, P)   \
    REGISTER_D_I_P(uint16_t, I, P)  \
    REGISTER_D_I_P(uint32_t, I, P)  \
    REGISTER_D_I_P(uint64_t, I, P)  \
    REGISTER_D_I_P(float32_t, I, P) \
    REGISTER_D_I_P(float64_t, I, P)

#define REGISTER_DS_IS_P(P)      \
    REGISTER_DS_I_P(int8_t, P)   \
    REGISTER_DS_I_P(int16_t, P)  \
    REGISTER_DS_I_P(int32_t, P)  \
    REGISTER_DS_I_P(int64_t, P)  \
    REGISTER_DS_I_P(uint8_t, P)  \
    REGISTER_DS_I_P(uint16_t, P) \
    REGISTER_DS_I_P(uint32_t, P) \
    REGISTER_DS_I_P(uint64_t, P)

    REGISTER_DS_IS_P(int32_t)
    REGISTER_DS_IS_P(int64_t)
    REGISTER_DS_IS_P(uint32_t)
    REGISTER_DS_IS_P(uint64_t)
}

}
