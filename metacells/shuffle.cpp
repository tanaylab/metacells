#include "metacells/extensions.h"

namespace metacells {

// TODO: Duplicated from relayout.
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
    WithoutGil without_gil{};
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
shuffle_dense(pybind11::array_t<D>& matrix_array, const size_t random_seed) {
    WithoutGil without_gil{};
    MatrixSlice<D> matrix(matrix_array, "matrix");

    parallel_loop(matrix.rows_count(), [&](size_t row_index) {
        size_t row_seed = random_seed == 0 ? 0 : random_seed + row_index * 997;
        shuffle_row(row_index, matrix, row_seed);
    });
}

void
register_shuffle(pybind11::module& module) {
#define REGISTER_D(D) module.def("shuffle_dense_" #D, &metacells::shuffle_dense<D>, "Shuffle dense matrix data.");

    REGISTER_D(int8_t)
    REGISTER_D(int16_t)
    REGISTER_D(int32_t)
    REGISTER_D(int64_t)
    REGISTER_D(uint8_t)
    REGISTER_D(uint16_t)
    REGISTER_D(uint32_t)
    REGISTER_D(uint64_t)
    REGISTER_D(float32_t)
    REGISTER_D(float64_t)

#define REGISTER_D_I_P(D, I, P)                         \
    module.def("shuffle_compressed_" #D "_" #I "_" #P,  \
               &metacells::shuffle_compressed<D, I, P>, \
               "Shuffle compressed matrix data.");

#define REGISTER_DS_I_P(I, P)       \
    REGISTER_D_I_P(bool, I, P)      \
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
