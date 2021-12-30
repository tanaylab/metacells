#include "metacells/extensions.h"

namespace metacells {

/// See the Python `metacell.tools.outlier_cells._collect_fold_factors` function.
template<typename D>
static void
fold_factor_dense(pybind11::array_t<D>& data_array,
                  const float64_t min_gene_fold_factor,
                  const bool abs_folds,
                  const pybind11::array_t<D>& total_of_rows_array,
                  const pybind11::array_t<D>& fraction_of_columns_array) {
    WithoutGil without_gil{};
    MatrixSlice<D> data(data_array, "data");
    ConstArraySlice<D> total_of_rows(total_of_rows_array, "total_of_rows");
    ConstArraySlice<D> fraction_of_columns(fraction_of_columns_array, "fraction_of_columns");

    FastAssertCompare(total_of_rows.size(), ==, data.rows_count());
    FastAssertCompare(fraction_of_columns.size(), ==, data.columns_count());

    const size_t rows_count = data.rows_count();
    const size_t columns_count = data.columns_count();
    if (abs_folds) {
        parallel_loop(rows_count, [&](size_t row_index) {
            const auto row_total = total_of_rows[row_index];
            auto row_data = data.get_row(row_index);
            for (size_t column_index = 0; column_index < columns_count; ++column_index) {
                const auto column_fraction = fraction_of_columns[column_index];
                const auto expected = row_total * column_fraction;
                auto& value = row_data[column_index];
                value = D(log((value + 1.0) / (expected + 1.0)) * LOG2_SCALE);
                if (abs(value) < min_gene_fold_factor) {
                    value = 0;
                }
            }
        });
    } else {
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
}

/// See the Python `metacell.tools.outlier_cells._collect_fold_factors` function.
template<typename D, typename I, typename P>
static void
fold_factor_compressed(pybind11::array_t<D>& data_array,
                       pybind11::array_t<I>& indices_array,
                       pybind11::array_t<P>& indptr_array,
                       const float64_t min_gene_fold_factor,
                       const bool abs_folds,
                       const pybind11::array_t<D>& total_of_bands_array,
                       const pybind11::array_t<D>& fraction_of_elements_array) {
    WithoutGil without_gil{};
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
        if (abs_folds) {
            for (size_t position = 0; position < band_elements_count; ++position) {
                const auto element_index = band_indices[position];
                const auto element_fraction = fraction_of_elements[element_index];
                const auto expected = band_total * element_fraction;
                auto& value = band_data[position];
                value = D(log((value + 1.0) / (expected + 1.0)) * LOG2_SCALE);
                if (abs(value) < min_gene_fold_factor) {
                    value = 0;
                }
            }
        } else {
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
        }
    });
}

template<typename D>
static void
collect_distinct_folds(ArraySlice<int32_t> gene_indices,
                       ArraySlice<float32_t> gene_folds,
                       ConstArraySlice<D> fold_in_cell,
                       bool abs_folds) {
    TmpVectorSizeT raii_indices;
    auto tmp_indices = raii_indices.array_slice("tmp_indices", fold_in_cell.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

    if (abs_folds) {
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
    } else {
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
    }

    for (size_t position = 0; position < gene_indices.size(); ++position) {
        size_t gene_index = tmp_indices[position];
        gene_indices[position] = int32_t(gene_index);
        gene_folds[position] = float32_t(fold_in_cell[gene_index]);
    }
}

template<typename D>
static void
top_distinct(pybind11::array_t<int32_t>& gene_indices_array,
             pybind11::array_t<float32_t>& gene_folds_array,
             const pybind11::array_t<D>& fold_in_cells_array,
             bool abs_folds) {
    WithoutGil without_gil{};
    MatrixSlice<float32_t> gene_folds(gene_folds_array, "gene_folds");
    MatrixSlice<int32_t> gene_indices(gene_indices_array, "gene_indices");
    ConstMatrixSlice<D> fold_in_cells(fold_in_cells_array, "fold_in_cells");

    size_t cells_count = fold_in_cells.rows_count();
    size_t genes_count = fold_in_cells.columns_count();
    size_t distinct_count = gene_indices.columns_count();

    FastAssertCompare(distinct_count, <, genes_count);
    FastAssertCompare(gene_indices.rows_count(), ==, cells_count);
    FastAssertCompare(gene_folds.rows_count(), ==, cells_count);
    FastAssertCompare(gene_folds.columns_count(), ==, distinct_count);

    parallel_loop(cells_count, [&](size_t cell_index) {
        collect_distinct_folds(gene_indices.get_row(cell_index),
                               gene_folds.get_row(cell_index),
                               fold_in_cells.get_row(cell_index),
                               abs_folds);
    });
}

void
register_folds(pybind11::module& module) {
#define REGISTER_F(F)                                                                                   \
    module.def("top_distinct_" #F, &metacells::top_distinct<F>, "Collect the topmost distinct genes."); \
    module.def("fold_factor_dense_" #F, &metacells::fold_factor_dense<F>, "Fold factors of dense data.");

    REGISTER_F(float32_t)
    REGISTER_F(float64_t)

#define REGISTER_F_I_P(F, I, P)                             \
    module.def("fold_factor_compressed_" #F "_" #I "_" #P,  \
               &metacells::fold_factor_compressed<F, I, P>, \
               "Fold factors of compressed data.");

#define REGISTER_FS_I_P(I, P)       \
    REGISTER_F_I_P(float32_t, I, P) \
    REGISTER_F_I_P(float64_t, I, P)

#define REGISTER_FS_IS_P(P)      \
    REGISTER_FS_I_P(int8_t, P)   \
    REGISTER_FS_I_P(int16_t, P)  \
    REGISTER_FS_I_P(int32_t, P)  \
    REGISTER_FS_I_P(int64_t, P)  \
    REGISTER_FS_I_P(uint8_t, P)  \
    REGISTER_FS_I_P(uint16_t, P) \
    REGISTER_FS_I_P(uint32_t, P) \
    REGISTER_FS_I_P(uint64_t, P)

    REGISTER_FS_IS_P(int32_t)
    REGISTER_FS_IS_P(int64_t)
    REGISTER_FS_IS_P(uint32_t)
    REGISTER_FS_IS_P(uint64_t)
}

}
