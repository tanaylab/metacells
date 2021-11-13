#include "metacells/extensions.h"

namespace metacells {

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
rank_rows(const pybind11::array_t<D>& input_matrix, pybind11::array_t<D>& output_array, const size_t rank) {
    WithoutGil without_gil{};
    ConstMatrixSlice<D> input(input_matrix, "input");
    ArraySlice<D> output(output_array, "array");

    const size_t rows_count = input.rows_count();
    FastAssertCompare(rows_count, ==, output_array.size());
    FastAssertCompare(rank, <, input.columns_count());

    parallel_loop(rows_count, [&](size_t row_index) { output[row_index] = rank_row_element(row_index, input, rank); });
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

    parallel_loop(rows_count, [&](size_t row_index) { rank_matrix_row(row_index, matrix, ascending); });
}

void
register_rank(pybind11::module& module) {
#define REGISTER_D(D)                                                                               \
    module.def("rank_rows_" #D, &metacells::rank_rows<D>, "Collect the rank element in each row."); \
    module.def("rank_matrix_" #D, &metacells::rank_matrix<D>, "Replace matrix data with ranks.");

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
}

}
