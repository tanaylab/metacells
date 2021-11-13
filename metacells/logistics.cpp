#include "metacells/extensions.h"

namespace metacells {

template<typename F>
static void
cross_logistics_dense(const pybind11::array_t<F>& first_input_array,
                      const pybind11::array_t<F>& second_input_array,
                      const float64_t location,
                      const float64_t slope,
                      pybind11::array_t<float32_t>& output_array) {
    WithoutGil without_gil{};
    ConstMatrixSlice<F> first_input(first_input_array, "input");
    ConstMatrixSlice<F> second_input(second_input_array, "input");
    MatrixSlice<float32_t> output(output_array, "output");

    const auto first_rows_count = first_input.rows_count();
    const auto second_rows_count = second_input.rows_count();
    FastAssertCompare(second_input.columns_count(), ==, first_input.columns_count());

    FastAssertCompare(output.rows_count(), ==, first_rows_count);
    FastAssertCompare(output.columns_count(), ==, second_rows_count);

    float64_t min_dist = float32_t(1.0 / (1.0 + exp(slope * location)));
    float64_t scale = 1.0 / (1.0 - min_dist);
    parallel_loop(first_rows_count, [&](size_t first_row_index) {
        auto output_row = output.get_row(first_row_index);
        const auto first_input_row = first_input.get_row(first_row_index);

        for (size_t second_row_index = 0; second_row_index < second_rows_count; ++second_row_index) {
            const float64_t logistic =
                logistics_two_dense_rows(first_input_row, second_input.get_row(second_row_index), location, slope);
            output_row[second_row_index] = float32_t((logistic - min_dist) * scale);
        }
    });
}

template<typename F>
static void
pairs_logistics_dense(const pybind11::array_t<F>& first_input_array,
                      const pybind11::array_t<F>& second_input_array,
                      const float64_t location,
                      const float64_t slope,
                      pybind11::array_t<float32_t>& output_array) {
    WithoutGil without_gil{};
    ConstMatrixSlice<F> first_input(first_input_array, "input");
    ConstMatrixSlice<F> second_input(second_input_array, "input");
    ArraySlice<float32_t> output(output_array, "output");

    const auto rows_count = first_input.rows_count();
    const auto columns_count = first_input.columns_count();

    FastAssertCompare(second_input.rows_count(), ==, rows_count);
    FastAssertCompare(second_input.columns_count(), ==, columns_count);
    FastAssertCompare(output.size(), ==, rows_count);

    float64_t min_dist = float32_t(1.0 / (1.0 + exp(slope * location)));
    float64_t scale = 1.0 / (1.0 - min_dist);
    parallel_loop(rows_count, [&](size_t row_index) {
        const float32_t logistic =
            logistics_two_dense_rows(first_input.get_row(row_index), second_input.get_row(row_index), location, slope);
        output[row_index] = float32_t((logistic - min_dist) * scale);
    });
}

template<typename F>
static float64_t
logistics_two_dense_rows(ConstArraySlice<F> first_row,
                         ConstArraySlice<F> second_row,
                         const float64_t location,
                         const float64_t slope) {
    FastAssertCompare(second_row.size(), ==, first_row.size());

    const size_t size = first_row.size();

    float64_t result = 0;

#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t index = 0; index < size; ++index) {
        float64_t diff = fabs(first_row[index] - second_row[index]);
        result += 1 / (1 + exp(slope * (location - diff)));
    }

    return result / size;
}

template<typename F>
static void
logistics_dense(const pybind11::array_t<F>& input_array,
                pybind11::array_t<float32_t>& output_array,
                const float64_t location,
                const float64_t slope) {
    WithoutGil without_gil{};
    ConstMatrixSlice<F> input(input_array, "input");
    MatrixSlice<float32_t> output(output_array, "output");

    const size_t rows_count = input.rows_count();

    FastAssertCompare(output.columns_count(), ==, rows_count);
    FastAssertCompare(output.rows_count(), ==, rows_count);

    const size_t iterations_count = (rows_count * (rows_count - 1)) / 2;

    for (size_t entry_index = 0; entry_index < rows_count; ++entry_index) {
        output.get_row(entry_index)[entry_index] = 0;
    }

    float64_t min_dist = float32_t(1.0 / (1.0 + exp(slope * location)));
    float64_t scale = 1.0 / (1.0 - min_dist);
    parallel_loop(iterations_count, [&](size_t iteration_index) {
        size_t some_index = iteration_index / (rows_count - 1);
        size_t other_index = iteration_index % (rows_count - 1);
        if (other_index < rows_count - 1 - some_index) {
            some_index = rows_count - 1 - some_index;
        } else {
            other_index = rows_count - 2 - other_index;
        }
        float64_t logistic =
            logistics_two_dense_rows(input.get_row(some_index), input.get_row(other_index), location, slope);
        logistic = (logistic - min_dist) * scale;
        output.get_row(some_index)[other_index] = float32_t(logistic);
        output.get_row(other_index)[some_index] = float32_t(logistic);
    });
}

void
register_logistics(pybind11::module& module) {
#define REGISTER_F(F)                                                                                       \
    module.def("logistics_dense_" #F, &metacells::logistics_dense<F>, "Correlate rows of dense matrices."); \
    module.def("cross_logistics_dense_" #F,                                                                 \
               &metacells::cross_logistics_dense<F>,                                                        \
               "Cross-logistics rows of dense matrices.");                                                  \
    module.def("pairs_logistics_dense_" #F,                                                                 \
               &metacells::pairs_logistics_dense<F>,                                                        \
               "Pairs-logistics rows of dense matrices.");

    REGISTER_F(float32_t)
    REGISTER_F(float64_t)
}

}
