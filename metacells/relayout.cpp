#include "metacells/extensions.h"

namespace metacells {

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

    for (size_t input_element_offset = start_input_element_offset; input_element_offset < stop_input_element_offset;
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

    for (size_t input_element_offset = start_input_element_offset; input_element_offset < stop_input_element_offset;
         ++input_element_offset) {
        auto input_element_index = input_indices[input_element_offset];
        auto input_element_data = input_data[input_element_offset];

        auto output_band_index = input_element_index;
        auto output_element_data = input_element_data;

        auto atomic_output_element_offset = reinterpret_cast<std::atomic<P>*>(&output_indptr[output_band_index]);
        auto output_element_offset = atomic_output_element_offset->fetch_add(1, std::memory_order_relaxed);

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

    parallel_loop(
        input_indptr.size() - 1,
        [&](size_t input_band_index) {
            parallel_collect_compressed_band(
                input_band_index, input_data, input_indices, input_indptr, output_data, output_indices, output_indptr);
        },
        [&](size_t input_band_index) {
            serial_collect_compressed_band(
                input_band_index, input_data, input_indices, input_indptr, output_data, output_indices, output_indptr);
        });
}

// TODO: Duplicated from shuffle.
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
    std::sort(tmp_positions.begin(), tmp_positions.end(), [&](const size_t left_position, const size_t right_position) {
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

typedef bool bool_t;

void
register_relayout(pybind11::module& module) {
#define REGISTER_D_I_P(D, I, P)                             \
    module.def("collect_compressed_" #D "_" #I "_" #P,      \
               &collect_compressed<D, I, P>,                \
               "Collect compressed data for relayout.");    \
    module.def("sort_compressed_indices_" #D "_" #I "_" #P, \
               &sort_compressed_indices<D, I, P>,           \
               "Sort indices in a compressed matrix.");

#define REGISTER_DS_I_P(I, P)       \
    REGISTER_D_I_P(bool_t, I, P)    \
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
