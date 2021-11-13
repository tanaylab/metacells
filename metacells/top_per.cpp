#include "metacells/extensions.h"

namespace metacells {

template<typename D>
static void
collect_top_row(const size_t row_index,
                const size_t degree,
                ConstMatrixSlice<D>& similarity_matrix,
                ArraySlice<int32_t> output_indices,
                ArraySlice<D> output_data,
                bool ranks) {
    const size_t columns_count = similarity_matrix.columns_count();
    const auto row_similarities = similarity_matrix.get_row(row_index);

    const size_t start_position = row_index * degree;
    const size_t stop_position = start_position + degree;

    TmpVectorSizeT raii_positions;
    auto tmp_positions = raii_positions.array_slice("tmp_positions", columns_count);
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);

    std::nth_element(tmp_positions.begin(),
                     tmp_positions.begin() + degree,
                     tmp_positions.end(),
                     [&](const size_t left_column_index, const size_t right_column_index) {
                         D left_similarity = row_similarities[left_column_index];
                         D right_similarity = row_similarities[right_column_index];
                         return left_similarity > right_similarity;
                     });

    auto row_data = output_data.slice(start_position, stop_position);
    auto row_indices = output_indices.slice(start_position, stop_position);
    std::copy(tmp_positions.begin(), tmp_positions.begin() + degree, row_indices.begin());
    std::sort(row_indices.begin(), row_indices.end());

    if (!ranks) {
#ifdef __INTEL_COMPILER
#    pragma simd
#endif
        for (size_t location = 0; location < degree; ++location) {
            size_t index = row_indices[location];
            row_data[location] = row_similarities[index];
        }

        return;
    }

    tmp_positions = tmp_positions.slice(0, degree);
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);
    std::sort(tmp_positions.begin(), tmp_positions.end(), [&](const size_t left_position, const size_t right_position) {
        D left_similarity = row_similarities[row_indices[left_position]];
        D right_similarity = row_similarities[row_indices[right_position]];
        return left_similarity < right_similarity;
    });

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t location = 0; location < degree; ++location) {
        size_t position = tmp_positions[location];
        row_data[position] = D(location + 1);
    }
}

/// See the Python `metacell.utilities.computation.top_per` function.
template<typename D>
static void
collect_top(const size_t degree,
            const pybind11::array_t<D>& input_similarity_matrix,
            pybind11::array_t<int32_t>& output_indices_array,
            pybind11::array_t<D>& output_data_array,
            bool ranks) {
    WithoutGil without_gil{};

    ConstMatrixSlice<D> similarity_matrix(input_similarity_matrix, "similarity_matrix");
    const size_t rows_count = similarity_matrix.rows_count();
    const size_t columns_count = similarity_matrix.columns_count();

    ArraySlice<int32_t> output_indices(output_indices_array, "output_indices");
    ArraySlice<D> output_data(output_data_array, "output_data");

    FastAssertCompare(0, <, degree);
    FastAssertCompare(degree, <, columns_count);

    FastAssertCompare(output_indices.size(), ==, degree * rows_count);
    FastAssertCompare(output_data.size(), ==, degree * rows_count);

    parallel_loop(rows_count, [&](size_t row_index) {
        collect_top_row(row_index, degree, similarity_matrix, output_indices, output_data, ranks);
    });
}

void
register_top_per(pybind11::module& module) {
#define REGISTER_D(D) module.def("collect_top_" #D, &collect_top<D>, "Collect the topmost elements.");

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
