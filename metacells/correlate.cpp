#include "metacells/extensions.h"

#ifdef USE_AVX2
#    include <immintrin.h>
#endif

namespace metacells {

struct Sums {
    float64_t values;
    float64_t squared;
};

template<typename F>
static Sums
sum_row_values(ConstArraySlice<F> input_row) {
    const F* const input_data = input_row.begin();
    const size_t columns_count = input_row.size();
    float64_t sum_values = 0;
    float64_t sum_squared = 0;
    for (size_t column_index = 0; column_index < columns_count; ++column_index) {
        const float64_t value = input_data[column_index];
        sum_values += value;
        sum_squared = fma(value, value, sum_squared);
    }
    return Sums{ sum_values, sum_squared };
}

template<typename F>
static float32_t
correlate_two_dense_rows(ConstArraySlice<F> some_values,
                         float64_t some_sum_values,
                         float64_t some_sum_squared,
                         ConstArraySlice<F> other_values,
                         float64_t other_sum_values,
                         float64_t other_sum_squared) {
    const size_t columns_count = some_values.size();
    const F* const some_values_data = some_values.begin();
    const F* const other_values_data = other_values.begin();

    float64_t both_sum_values = 0;
    size_t column_index = 0;
#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (; column_index < columns_count; ++column_index) {
        const float64_t some_values = float64_t(some_values_data[column_index]);
        const float64_t other_values = float64_t(other_values_data[column_index]);
        both_sum_values = fma(some_values, other_values, both_sum_values);
    }

    float64_t correlation = columns_count * both_sum_values - some_sum_values * other_sum_values;
    float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
    float64_t other_factor = columns_count * other_sum_squared - other_sum_values * other_sum_values;
    float64_t both_factors = sqrt(some_factor * other_factor);
    if (both_factors != 0) {
        correlation /= both_factors;
        return std::max(std::min(float32_t(correlation), float32_t(1.0)), float32_t(-1.0));
    } else {
        return 0.0;
    }
}

#ifdef USE_AVX2
template<>
float32_t
correlate_two_dense_rows(ConstArraySlice<float64_t> some_values,
                         float64_t some_sum_values,
                         float64_t some_sum_squared,
                         ConstArraySlice<float64_t> other_values,
                         float64_t other_sum_values,
                         float64_t other_sum_squared) {
    const size_t columns_count = some_values.size();
    const float64_t* const some_values_data = some_values.begin();
    const float64_t* const other_values_data = other_values.begin();

    float64_t both_sum_values = 0;
    size_t column_index = 0;
    size_t avx2_columns_count = (columns_count / 4) * 4;
    __m256d both_sum_avx2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    for (; column_index < avx2_columns_count; column_index += 4) {
        __m256d some_values_avx2 = _mm256_loadu_pd(some_values_data + column_index);
        __m256d other_values_avx2 = _mm256_loadu_pd(other_values_data + column_index);
        both_sum_avx2 = _mm256_fmadd_pd(some_values_avx2, other_values_avx2, both_sum_avx2);
    }
    both_sum_values = *(float64_t*)&both_sum_avx2[0] + *(float64_t*)&both_sum_avx2[1] + *(float64_t*)&both_sum_avx2[2]
                      + *(float64_t*)&both_sum_avx2[3];
#    ifdef __INTEL_COMPILER
#        pragma simd
#    endif
    for (; column_index < columns_count; ++column_index) {
        const float64_t some_values = float64_t(some_values_data[column_index]);
        const float64_t other_values = float64_t(other_values_data[column_index]);
        both_sum_values = fma(some_values, other_values, both_sum_values);
    }

    float64_t correlation = columns_count * both_sum_values - some_sum_values * other_sum_values;
    float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
    float64_t other_factor = columns_count * other_sum_squared - other_sum_values * other_sum_values;
    float64_t both_factors = sqrt(some_factor * other_factor);
    if (both_factors != 0) {
        correlation /= both_factors;
        return std::max(std::min(float32_t(correlation), float32_t(1.0)), float32_t(-1.0));
    } else {
        return 0.0;
    }
}

template<>
float32_t
correlate_two_dense_rows(ConstArraySlice<float32_t> some_values,
                         float64_t some_sum_values,
                         float64_t some_sum_squared,
                         ConstArraySlice<float32_t> other_values,
                         float64_t other_sum_values,
                         float64_t other_sum_squared) {
    const size_t columns_count = some_values.size();
    const float32_t* const some_values_data = some_values.begin();
    const float32_t* const other_values_data = other_values.begin();

    size_t column_index = 0;
    size_t avx2_columns_count = (columns_count / 8) * 8;
    __m256 both_sum_avx2 = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (; column_index < avx2_columns_count; column_index += 8) {
        __m256 some_values_avx2 = _mm256_loadu_ps(some_values_data + column_index);
        __m256 other_values_avx2 = _mm256_loadu_ps(other_values_data + column_index);
        both_sum_avx2 = _mm256_fmadd_ps(some_values_avx2, other_values_avx2, both_sum_avx2);
    }
    float64_t both_sum_values = *(float32_t*)&both_sum_avx2[0] + *(float32_t*)&both_sum_avx2[1]
                                + *(float32_t*)&both_sum_avx2[2] + *(float32_t*)&both_sum_avx2[3]
                                + *(float32_t*)&both_sum_avx2[4] + *(float32_t*)&both_sum_avx2[5]
                                + *(float32_t*)&both_sum_avx2[6] + *(float32_t*)&both_sum_avx2[7];
#    ifdef __INTEL_COMPILER
#        pragma simd
#    endif
    for (; column_index < columns_count; ++column_index) {
        const float32_t some_values = some_values_data[column_index];
        const float32_t other_values = other_values_data[column_index];
        both_sum_values = fma(some_values, other_values, both_sum_values);
    }

    float64_t correlation = columns_count * both_sum_values - some_sum_values * other_sum_values;
    float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
    float64_t other_factor = columns_count * other_sum_squared - other_sum_values * other_sum_values;
    float64_t both_factors = sqrt(some_factor * other_factor);
    if (both_factors != 0) {
        correlation /= both_factors;
        return std::max(std::min(float32_t(correlation), float32_t(1.0)), float32_t(-1.0));
    } else {
        return 0.0;
    }
}
#endif

#define MANY_ROWS 2

struct ManyCorrelations {
    float64_t correlations[MANY_ROWS];
};

template<typename F>
static ManyCorrelations
correlate_many_dense_rows(const F* const some_values_data,
                          ConstMatrixSlice<F> values,
                          const float64_t some_sum_values,
                          const float64_t some_sum_squared,
                          const std::vector<float64_t>& row_sum_values,
                          const std::vector<float64_t>& row_sum_squared,
                          const size_t other_begin_index) {
    const size_t columns_count = values.columns_count();

    const F* other_values_data[MANY_ROWS];
    F both_sum_values[MANY_ROWS];
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        other_values_data[which_other] = values.get_row(other_begin_index + which_other).begin();
        both_sum_values[which_other] = 0;
    }
#ifdef __INTEL_COMPILER
#    pragma simd
#endif
    for (size_t column_index = 0; column_index < columns_count; ++column_index) {
        const F some_value = some_values_data[column_index];
        for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
            both_sum_values[which_other] += some_value * other_values_data[which_other][column_index];
        }
    }

    ManyCorrelations results;
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        const float64_t other_sum_values = row_sum_values[other_begin_index + which_other];
        const float64_t other_sum_squared = row_sum_squared[other_begin_index + which_other];
        results.correlations[which_other] =
            columns_count * float64_t(both_sum_values[which_other]) - some_sum_values * other_sum_values;
        const float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
        const float64_t other_factor = columns_count * other_sum_squared - other_sum_values * other_sum_values;
        const float64_t both_factors = sqrt(some_factor * other_factor);
        if (both_factors != 0) {
            results.correlations[which_other] /= both_factors;
            results.correlations[which_other] = std::max(std::min(results.correlations[which_other], 1.0), -1.0);
        } else {
            results.correlations[which_other] = 0.0;
        }
    }

    return results;
}

#ifdef USE_AVX2
template<>
ManyCorrelations
correlate_many_dense_rows(const float64_t* const some_values_data,
                          ConstMatrixSlice<float64_t> values,
                          const float64_t some_sum_values,
                          const float64_t some_sum_squared,
                          const std::vector<float64_t>& row_sum_values,
                          const std::vector<float64_t>& row_sum_squared,
                          const size_t other_begin_index) {
    const size_t columns_count = values.columns_count();

    const float64_t* other_values_data[MANY_ROWS];
    __m256d both_sum_avx2[MANY_ROWS];
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        other_values_data[which_other] = values.get_row(other_begin_index + which_other).begin();
        both_sum_avx2[which_other] = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    }

    size_t column_index = 0;
    size_t avx2_columns_count = (columns_count / 4) * 4;
    for (; column_index < avx2_columns_count; column_index += 4) {
        __m256d some_values_avx2 = _mm256_loadu_pd(some_values_data + column_index);
        for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
            __m256d other_values_avx2 = _mm256_loadu_pd(other_values_data[which_other] + column_index);
            both_sum_avx2[which_other] =
                _mm256_fmadd_pd(some_values_avx2, other_values_avx2, both_sum_avx2[which_other]);
        }
    }
    float64_t both_sum_values[MANY_ROWS];
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        both_sum_values[which_other] =
            *(float64_t*)&both_sum_avx2[which_other][0] + *(float64_t*)&both_sum_avx2[which_other][1]
            + *(float64_t*)&both_sum_avx2[which_other][2] + *(float64_t*)&both_sum_avx2[which_other][3];
    }
#    ifdef __INTEL_COMPILER
#        pragma simd
#    endif
    for (; column_index < columns_count; ++column_index) {
        const float64_t some_value = some_values_data[column_index];
        for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
            both_sum_values[which_other] += some_value * other_values_data[which_other][column_index];
        }
    }

    ManyCorrelations results;
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        const float64_t other_sum_values = row_sum_values[other_begin_index + which_other];
        const float64_t other_sum_squared = row_sum_squared[other_begin_index + which_other];
        results.correlations[which_other] =
            columns_count * float64_t(both_sum_values[which_other]) - some_sum_values * other_sum_values;
        const float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
        const float64_t other_factor = columns_count * other_sum_squared - other_sum_values * other_sum_values;
        const float64_t both_factors = sqrt(some_factor * other_factor);
        if (both_factors != 0) {
            results.correlations[which_other] /= both_factors;
            results.correlations[which_other] = std::max(std::min(results.correlations[which_other], 1.0), -1.0);
        } else {
            results.correlations[which_other] = 0.0;
        }
    }

    return results;
}

template<>
ManyCorrelations
correlate_many_dense_rows(const float32_t* const some_values_data,
                          ConstMatrixSlice<float32_t> values,
                          const float64_t some_sum_values,
                          const float64_t some_sum_squared,
                          const std::vector<float64_t>& row_sum_values,
                          const std::vector<float64_t>& row_sum_squared,
                          const size_t other_begin_index) {
    const size_t columns_count = values.columns_count();

    const float32_t* other_values_data[MANY_ROWS];
    __m256 both_sum_avx2[MANY_ROWS];
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        other_values_data[which_other] = values.get_row(other_begin_index + which_other).begin();
        both_sum_avx2[which_other] = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    size_t column_index = 0;
    size_t avx2_columns_count = (columns_count / 8) * 8;
    for (; column_index < avx2_columns_count; column_index += 8) {
        __m256 some_values_avx2 = _mm256_loadu_ps(some_values_data + column_index);
        for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
            __m256 other_values_avx2 = _mm256_loadu_ps(other_values_data[which_other] + column_index);
            both_sum_avx2[which_other] =
                _mm256_fmadd_ps(some_values_avx2, other_values_avx2, both_sum_avx2[which_other]);
        }
    }
    float64_t both_sum_values[MANY_ROWS];
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        both_sum_values[which_other] =
            *(float32_t*)&both_sum_avx2[which_other][0] + *(float32_t*)&both_sum_avx2[which_other][1]
            + *(float32_t*)&both_sum_avx2[which_other][2] + *(float32_t*)&both_sum_avx2[which_other][3]
            + *(float32_t*)&both_sum_avx2[which_other][4] + *(float32_t*)&both_sum_avx2[which_other][5]
            + *(float32_t*)&both_sum_avx2[which_other][6] + *(float32_t*)&both_sum_avx2[which_other][7];
    }
#    ifdef __INTEL_COMPILER
#        pragma simd
#    endif
    for (; column_index < columns_count; ++column_index) {
        const float32_t some_value = some_values_data[column_index];
        for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
            both_sum_values[which_other] += some_value * other_values_data[which_other][column_index];
        }
    }

    ManyCorrelations results;
    for (size_t which_other = 0; which_other < MANY_ROWS; ++which_other) {
        const float64_t other_sum_values = row_sum_values[other_begin_index + which_other];
        const float64_t other_sum_squared = row_sum_squared[other_begin_index + which_other];
        results.correlations[which_other] =
            columns_count * float64_t(both_sum_values[which_other]) - some_sum_values * other_sum_values;
        const float64_t some_factor = columns_count * some_sum_squared - some_sum_values * some_sum_values;
        const float64_t other_factor = columns_count * other_sum_squared - other_sum_values * other_sum_values;
        const float64_t both_factors = sqrt(some_factor * other_factor);
        if (both_factors != 0) {
            results.correlations[which_other] /= both_factors;
            results.correlations[which_other] = std::max(std::min(results.correlations[which_other], 1.0), -1.0);
        } else {
            results.correlations[which_other] = 0.0;
        }
    }

    return results;
}
#endif

static size_t
unrolled_iterations_count(const size_t rows_count, const size_t unroll_size) {
    const size_t full_rows_groups_count = (rows_count - 1) / unroll_size;
    const size_t full_rows_groups_sum = (full_rows_groups_count * (full_rows_groups_count + 1)) / 2;
    const size_t last_rows_count = (rows_count - 1) % unroll_size;
    const size_t last_rows_size = size_t(ceil((rows_count - 1.0) / unroll_size));
    const size_t iterations_count = full_rows_groups_sum * unroll_size + last_rows_count * last_rows_size;
    return iterations_count;
}

template<typename F>
static void
correlate_dense(const pybind11::array_t<F>& input_array, pybind11::array_t<float32_t>& output_array) {
    WithoutGil without_gil{};
    ConstMatrixSlice<F> input(input_array, "input");
    MatrixSlice<float32_t> output(output_array, "output");

    const auto rows_count = input.rows_count();

    FastAssertCompare(output.rows_count(), ==, input.rows_count());
    FastAssertCompare(output.columns_count(), ==, input.rows_count());

    TmpVectorFloat64 row_sum_values_raii;
    auto row_sum_values = row_sum_values_raii.vector(rows_count);

    TmpVectorFloat64 row_sum_squared_raii;
    auto row_sum_squared = row_sum_squared_raii.vector(rows_count);

    parallel_loop(rows_count, [&](size_t row_index) {
        const auto sums = sum_row_values(input.get_row(row_index));
        row_sum_values[row_index] = sums.values;
        row_sum_squared[row_index] = sums.squared;
    });

    for (size_t entry_index = 0; entry_index < rows_count; ++entry_index) {
        output.get_row(entry_index)[entry_index] = 1.0;
    }

    const size_t unroll_size = MANY_ROWS;
    const size_t iterations_count = unrolled_iterations_count(rows_count, unroll_size);

    parallel_loop(iterations_count, [&](size_t iteration_index) {
        size_t min_rows_count =
            size_t(round((sqrt(unroll_size * (unroll_size + 8.0 * iteration_index)) - unroll_size + 1.0) / 2.0));
        size_t min_rows_iterations_count = unrolled_iterations_count(min_rows_count, unroll_size);

        while (min_rows_count > 1 and min_rows_iterations_count > iteration_index) {
            min_rows_count -= 1;
            min_rows_iterations_count = unrolled_iterations_count(min_rows_count, unroll_size);
        }

        while (true) {
            const size_t up_min_rows_iterations_count = unrolled_iterations_count(min_rows_count + 1, unroll_size);
            if (up_min_rows_iterations_count > iteration_index) {
                break;
            }
            ++min_rows_count;
            min_rows_iterations_count = up_min_rows_iterations_count;
        }

        const size_t some_index = min_rows_count;
        const size_t extra_iterations = iteration_index - min_rows_iterations_count;
        const size_t other_begin_index = extra_iterations * unroll_size;
        const size_t other_end_index = std::min(other_begin_index + unroll_size, some_index);

        if (other_begin_index + MANY_ROWS == other_end_index) {
            const ManyCorrelations results = correlate_many_dense_rows(input.get_row(some_index).begin(),
                                                                       input,
                                                                       row_sum_values[some_index],
                                                                       row_sum_squared[some_index],
                                                                       row_sum_values,
                                                                       row_sum_squared,
                                                                       other_begin_index);
            for (int which_other = 0; which_other < MANY_ROWS; ++which_other) {
                const size_t other_index = other_begin_index + which_other;
                output.get_row(some_index)[other_index] = results.correlations[which_other];
                output.get_row(other_index)[some_index] = results.correlations[which_other];
            }
        } else {
            for (size_t other_index = other_begin_index; other_index != other_end_index; ++other_index) {
                float32_t correlation = correlate_two_dense_rows(input.get_row(some_index),
                                                                 row_sum_values[some_index],
                                                                 row_sum_squared[some_index],
                                                                 input.get_row(other_index),
                                                                 row_sum_values[other_index],
                                                                 row_sum_squared[other_index]);
                output.get_row(some_index)[other_index] = correlation;
                output.get_row(other_index)[some_index] = correlation;
            }
        }
    });
}

template<typename F>
static void
cross_correlate_dense(const pybind11::array_t<F>& first_input_array,
                      const pybind11::array_t<F>& second_input_array,
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

    TmpVectorFloat64 second_row_sum_values_raii;
    auto second_row_sum_values = second_row_sum_values_raii.vector(second_rows_count);

    TmpVectorFloat64 second_row_sum_squared_raii;
    auto second_row_sum_squared = second_row_sum_squared_raii.vector(second_rows_count);

    parallel_loop(second_rows_count, [&](size_t second_row_index) {
        const auto sums = sum_row_values(second_input.get_row(second_row_index));
        second_row_sum_values[second_row_index] = sums.values;
        second_row_sum_squared[second_row_index] = sums.squared;
    });

    parallel_loop(first_rows_count, [&](size_t first_row_index) {
        auto output_row = output.get_row(first_row_index);
        const auto first_input_row = first_input.get_row(first_row_index);
        const auto sums = sum_row_values(first_input_row);
        const float64_t first_row_sum_values = sums.values;
        const float64_t first_row_sum_squared = sums.squared;

        size_t second_row_index = 0;
        while (second_row_index < second_rows_count) {
            if (second_row_index + MANY_ROWS <= second_rows_count) {
                const ManyCorrelations results = correlate_many_dense_rows(first_input_row.begin(),
                                                                           second_input,
                                                                           first_row_sum_values,
                                                                           first_row_sum_squared,
                                                                           second_row_sum_values,
                                                                           second_row_sum_squared,
                                                                           second_row_index);
                for (int which_other = 0; which_other < MANY_ROWS; ++which_other) {
                    output_row[second_row_index] = results.correlations[which_other];
                    ++second_row_index;
                }
            } else {
                output_row[second_row_index] = correlate_two_dense_rows(first_input_row,
                                                                        first_row_sum_values,
                                                                        first_row_sum_squared,
                                                                        second_input.get_row(second_row_index),
                                                                        second_row_sum_values[second_row_index],
                                                                        second_row_sum_squared[second_row_index]);
                ++second_row_index;
            }
        }
    });
}

template<typename F>
static void
pairs_correlate_dense(const pybind11::array_t<F>& first_input_array,
                      const pybind11::array_t<F>& second_input_array,
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

    parallel_loop(rows_count, [&](size_t row_index) {
        const auto first_input_row = first_input.get_row(row_index);
        const auto second_input_row = second_input.get_row(row_index);

        const auto first_sums = sum_row_values(first_input_row);
        const auto second_sums = sum_row_values(second_input_row);

        output[row_index] = correlate_two_dense_rows(first_input_row,
                                                     first_sums.values,
                                                     first_sums.squared,
                                                     second_input_row,
                                                     second_sums.values,
                                                     second_sums.squared);
    });
}

void
register_correlate(pybind11::module& module) {
#define REGISTER_F(F)                                                                                       \
    module.def("correlate_dense_" #F, &metacells::correlate_dense<F>, "Correlate rows of dense matrices."); \
    module.def("cross_correlate_dense_" #F,                                                                 \
               &metacells::cross_correlate_dense<F>,                                                        \
               "Cross-correlate rows of dense matrices.");                                                  \
    module.def("pairs_correlate_dense_" #F,                                                                 \
               &metacells::pairs_correlate_dense<F>,                                                        \
               "Pairs-correlate rows of dense matrices.");

    REGISTER_F(float32_t)
    REGISTER_F(float64_t)
}

}
