#include "metacells/extensions.h"

namespace metacells {

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
    WithoutGil without_gil{};
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
    WithoutGil without_gil{};
    ConstCompressedMatrix<D, I, P> values(ConstArraySlice<D>(values_data_array, "values_data"),
                                          ConstArraySlice<I>(values_indices_array, "values_indices"),
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

void
register_auroc(pybind11::module& module) {
#define REGISTER_D(D) \
    module.def("auroc_dense_matrix_" #D, &metacells::auroc_dense_matrix<D>, "AUROC for dense matrix.");

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

#define REGISTER_D_I_P(D, I, P)                              \
    module.def("auroc_compressed_matrix_" #D "_" #I "_" #P,  \
               &metacells::auroc_compressed_matrix<D, I, P>, \
               "AUROC for compressed matrix.");

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
