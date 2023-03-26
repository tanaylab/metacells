#include "metacells/extensions.h"

namespace metacells {

/// See the Python `metacell.tools.deviants.find_deviant_cells` function.
static void
compute_cell_gaps(const pybind11::array_t<float32_t>& umis_per_gene_per_cell_array,
                  const pybind11::array_t<float32_t>& fraction_per_gene_per_cell_array,
                  const pybind11::array_t<float32_t>& log_fraction_per_gene_per_cell_array,
                  const pybind11::array_t<int32_t>& candidate_index_per_cell_array,
                  const pybind11::array_t<bool>& deviant_per_cell_array,
                  const pybind11::array_t<bool>& active_per_cell_array,
                  const pybind11::array_t<float32_t>& min_gap_per_gene_array,
                  const size_t candidates_count,
                  const size_t gap_skip_cells,
                  const size_t max_deviant_cells_count,
                  const float32_t max_deviant_cells_fraction,
                  const float32_t min_compare_umis,
                  pybind11::array_t<float32_t>& max_gap_per_cell_array) {
    WithoutGil without_gil{};

    ConstMatrixSlice<float32_t> umis_per_gene_per_cell(umis_per_gene_per_cell_array, "umis_per_gene_per_cell");
    ConstMatrixSlice<float32_t> fraction_per_gene_per_cell(fraction_per_gene_per_cell_array,
                                                           "fraction_per_gene_per_cell");
    ConstMatrixSlice<float32_t> log_fraction_per_gene_per_cell(log_fraction_per_gene_per_cell_array,
                                                               "log_fraction_per_gene_per_cell");
    ConstArraySlice<int32_t> candidate_index_per_cell(candidate_index_per_cell_array, "candidate_index_per_cell");
    ConstArraySlice<bool> deviant_per_cell(deviant_per_cell_array, "deviant_per_cell");
    ConstArraySlice<bool> active_per_cell(active_per_cell_array, "active_per_cell");
    ConstArraySlice<float32_t> min_gap_per_gene(min_gap_per_gene_array, "min_gap_per_gene");
    ArraySlice<float32_t> max_gap_per_cell(max_gap_per_cell_array, "max_gap_per_cell");

    const size_t cells_count = umis_per_gene_per_cell.rows_count();
    const size_t genes_count = umis_per_gene_per_cell.columns_count();
    FastAssertCompare(fraction_per_gene_per_cell.rows_count(), ==, cells_count);
    FastAssertCompare(fraction_per_gene_per_cell.columns_count(), ==, genes_count);
    FastAssertCompare(log_fraction_per_gene_per_cell.rows_count(), ==, cells_count);
    FastAssertCompare(log_fraction_per_gene_per_cell.columns_count(), ==, genes_count);
    FastAssertCompare(candidate_index_per_cell.size(), ==, cells_count);
    FastAssertCompare(min_gap_per_gene.size(), ==, genes_count);
    FastAssertCompare(max_gap_per_cell.size(), ==, cells_count);
    FastAssertCompare(gap_skip_cells, >=, 1);
    FastAssertCompare(gap_skip_cells, <=, 3);

    parallel_loop(candidates_count, [&](size_t candidate_index) {
        TmpVectorSizeT cell_indices_raii;
        auto cell_index_of_position = cell_indices_raii.vector(cells_count);
        cell_index_of_position.clear();
        bool active_candidate = false;
        for (size_t cell_index = 0; cell_index < cells_count; ++cell_index) {
            if (size_t(candidate_index_per_cell[cell_index]) == candidate_index) {
                if (active_per_cell[cell_index]) {
                    active_candidate = true;
                }
                if (!deviant_per_cell[cell_index]) {
                    cell_index_of_position.push_back(cell_index);
                }
            }
        }
        const size_t candidate_cells_count = cell_index_of_position.size();
        if (!active_candidate || candidate_cells_count < 4) {
            return;
        }

        const size_t max_deviant_cells_count_of_candidate =
            std::min(candidate_cells_count - gap_skip_cells,
                     std::max(max_deviant_cells_count,
                              size_t(max_deviant_cells_fraction * candidate_cells_count + 0.5)));

        for (size_t gene_index = 0; gene_index < genes_count; ++gene_index) {
            const float32_t min_gap_of_gene = min_gap_per_gene[gene_index];

            bool has_significant_umis = false;
            for (size_t cell_position = 0; cell_position < candidate_cells_count; ++cell_position) {
                const auto umis = umis_per_gene_per_cell(cell_index_of_position[cell_position], gene_index);
                if (umis * 2 + 1e-6 > min_compare_umis) {
                    has_significant_umis = true;
                    break;
                }
            }

            if (!has_significant_umis) {
                continue;
            }

            // TODO: It may be faster to partition to low, middle and high regions and only sort the low and high
            // regions; fully sort only if the middle region is empty.
            std::sort(cell_index_of_position.begin(),
                      cell_index_of_position.end(),
                      [&](const size_t left_cell_index, const size_t right_cell_index) {
                          const auto left_fraction = log_fraction_per_gene_per_cell(left_cell_index, gene_index);
                          const auto right_fraction = log_fraction_per_gene_per_cell(right_cell_index, gene_index);
                          if (left_fraction < right_fraction) {
                              return true;
                          }
                          if (left_fraction > right_fraction) {
                              return false;
                          }
                          return log_fraction_per_gene_per_cell(left_cell_index, gene_index)
                                 < log_fraction_per_gene_per_cell(right_cell_index, gene_index);
                      });

            const size_t max_cell_position =
                std::min((candidate_cells_count - 1) / 2, max_deviant_cells_count_of_candidate);

            for (size_t cell_position = 0; cell_position < max_cell_position; ++cell_position) {
                const size_t first_cell_index = cell_index_of_position[cell_position];
                const size_t second_cell_index = cell_index_of_position[cell_position + gap_skip_cells];

                const float32_t first_cell_umis = umis_per_gene_per_cell(first_cell_index, gene_index);
                const float32_t second_cell_umis = umis_per_gene_per_cell(second_cell_index, gene_index);
                if (first_cell_umis + second_cell_umis + 1e-6 < min_compare_umis) {
                    continue;
                }

                const float32_t first_cell_log_fraction = log_fraction_per_gene_per_cell(first_cell_index, gene_index);
                const float32_t second_cell_log_fraction =
                    log_fraction_per_gene_per_cell(second_cell_index, gene_index);
                const float32_t full_gap = second_cell_log_fraction - first_cell_log_fraction - min_gap_of_gene;
                if (full_gap < 0.0) {
                    continue;
                }

                size_t stop_cell_position = cell_position + 1;
                if (gap_skip_cells > 1) {
                    const size_t middle_cell_index = cell_index_of_position[cell_position + 1];
                    const float32_t middle_cell_log_fraction =
                        log_fraction_per_gene_per_cell(middle_cell_index, gene_index);
                    const float32_t start_gap =
                        2.0 * (middle_cell_log_fraction - first_cell_log_fraction) - min_gap_of_gene;

                    if (start_gap < full_gap) {
                        stop_cell_position += 1;
                    }
                }

                for (size_t gap_cell_position = 0; gap_cell_position <= stop_cell_position; ++gap_cell_position) {
                    const size_t gap_cell_index = cell_index_of_position[gap_cell_position];
                    max_gap_per_cell[gap_cell_index] = std::max(max_gap_per_cell[gap_cell_index], full_gap);
                }
            }

            for (size_t cell_offset = 0; cell_offset < max_cell_position; ++cell_offset) {
                const size_t cell_position = candidate_cells_count - 1 - cell_offset - gap_skip_cells;
                const size_t first_cell_index = cell_index_of_position[cell_position];
                const size_t second_cell_index = cell_index_of_position[cell_position + gap_skip_cells];

                const float32_t first_cell_umis = umis_per_gene_per_cell(first_cell_index, gene_index);
                const float32_t second_cell_umis = umis_per_gene_per_cell(second_cell_index, gene_index);
                if (first_cell_umis + second_cell_umis < min_compare_umis) {
                    continue;
                }

                const float32_t first_cell_log_fraction = log_fraction_per_gene_per_cell(first_cell_index, gene_index);
                const float32_t second_cell_log_fraction =
                    log_fraction_per_gene_per_cell(second_cell_index, gene_index);

                const float32_t full_gap = second_cell_log_fraction - first_cell_log_fraction - min_gap_of_gene;
                if (full_gap <= 0.0) {
                    continue;
                }

                size_t start_cell_position = cell_position + 1;

                if (gap_skip_cells > 1) {
                    const size_t middle_cell_index = cell_index_of_position[cell_position + 1];
                    const float32_t middle_cell_log_fraction =
                        log_fraction_per_gene_per_cell(middle_cell_index, gene_index);
                    const float32_t start_gap =
                        2.0 * (middle_cell_log_fraction - first_cell_log_fraction) - min_gap_of_gene;

                    if (start_gap < full_gap) {
                        start_cell_position += 1;
                    }
                }

                for (size_t gap_cell_position = start_cell_position; gap_cell_position < candidate_cells_count;
                     ++gap_cell_position) {
                    const size_t gap_cell_index = cell_index_of_position[gap_cell_position];
                    max_gap_per_cell[gap_cell_index] = std::max(max_gap_per_cell[gap_cell_index], full_gap);
                }
            }
        }
    });
}

void
register_gaps(pybind11::module& module) {
    module.def("compute_cell_gaps",
               &metacells::compute_cell_gaps,
               "Compute the fold gaps between genes for each cell.");
}
}
