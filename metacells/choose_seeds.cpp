#include "metacells/extensions.h"

namespace metacells {

static std::vector<std::vector<int32_t>>
collect_connected_nodes(ConstCompressedMatrix<float32_t, int32_t, int32_t>& incoming_weights,
                        ConstArraySlice<int32_t> seed_of_nodes) {
    const size_t nodes_count = seed_of_nodes.size();
    std::vector<std::vector<int32_t>> connected_nodes(nodes_count);

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        if (seed_of_nodes[node_index] >= 0) {
            continue;
        }
        auto node_incoming = incoming_weights.get_band_indices(node_index);
        auto& node_connected = connected_nodes[node_index];
        node_connected.resize(node_incoming.size());
        auto copied = std::copy_if(node_incoming.begin(),
                                   node_incoming.end(),
                                   node_connected.begin(),
                                   [&](int32_t other_node_index) { return seed_of_nodes[other_node_index] < 0; });
        node_connected.resize(copied - node_connected.begin());
    }

    return connected_nodes;
}

static size_t
choose_seed_node(const std::vector<size_t>& tmp_candidates,
                 const std::vector<std::vector<int32_t>>& connected_nodes,
                 const float32_t min_seed_size_quantile,
                 const float32_t max_seed_size_quantile,
                 std::minstd_rand& random) {
    size_t size = tmp_candidates.size();

    TmpVectorSizeT raii_positions;
    auto tmp_positions = raii_positions.array_slice("tmp_positions", size);
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);

    size_t min_seed_rank = size_t(floor((size - 1) * min_seed_size_quantile));
    size_t max_seed_rank = size_t(ceil((size - 1) * max_seed_size_quantile));
    FastAssertCompare(0, <=, min_seed_rank);
    FastAssertCompare(min_seed_rank, <=, max_seed_rank);
    FastAssertCompare(max_seed_rank, <=, size - 1);

    std::nth_element(tmp_positions.begin(),
                     tmp_positions.begin() + min_seed_rank,
                     tmp_positions.end(),
                     [&](const size_t left_position, const size_t right_position) {
                         const auto left_node_index = tmp_candidates[left_position];
                         const auto right_node_index = tmp_candidates[right_position];
                         const auto left_size = connected_nodes[left_node_index].size();
                         const auto right_size = connected_nodes[right_node_index].size();
                         return left_size < right_size;
                     });

    std::nth_element(tmp_positions.begin() + min_seed_rank,
                     tmp_positions.begin() + max_seed_rank,
                     tmp_positions.end(),
                     [&](const size_t left_position, const size_t right_position) {
                         const auto left_node_index = tmp_candidates[left_position];
                         const auto right_node_index = tmp_candidates[right_position];
                         const auto left_size = connected_nodes[left_node_index].size();
                         const auto right_size = connected_nodes[right_node_index].size();
                         return left_size < right_size;
                     });

    const size_t selected = random() % (max_seed_rank + 1 - min_seed_rank);
    const size_t position = tmp_positions[min_seed_rank + selected];
    size_t seed_node_index = tmp_candidates[position];

    LOCATED_LOG(false)                                           //
        << " node: " << seed_node_index                          //
        << " size: " << connected_nodes[seed_node_index].size()  //
        << std::endl;
    return seed_node_index;
}

template<typename T>
static void
remove_sorted(std::vector<T>& vector, T value) {
    auto position = std::lower_bound(vector.begin(), vector.end(), value);
    if (position != vector.end() && *position == value) {
        vector.erase(position);
    } else {
        LOCATED_LOG(true) << " OOPS! removing nonexistent value" << std::endl;
        assert(false);
    }
}

static void
store_seed_node(ConstCompressedMatrix<float32_t, int32_t, int32_t>& outgoing_weights,
                ConstCompressedMatrix<float32_t, int32_t, int32_t>& incoming_weights,
                const size_t seed_index,
                const size_t seed_node_index,
                std::vector<size_t>& tmp_candidates,
                std::vector<std::vector<int32_t>>& connected_nodes,
                ArraySlice<int32_t> seed_of_nodes,
                const size_t seed_size) {
    remove_sorted(tmp_candidates, seed_node_index);

    auto seed_incoming_nodes = incoming_weights.get_band_indices(seed_node_index);
    auto seed_incoming_weights = incoming_weights.get_band_data(seed_node_index);

    TmpVectorSizeT positions_raii;
    auto tmp_positions = positions_raii.vector(seed_incoming_nodes.size());
    std::iota(tmp_positions.begin(), tmp_positions.end(), 0);

    tmp_positions.erase(std::remove_if(tmp_positions.begin(),
                                       tmp_positions.end(),
                                       [&](size_t position) {
                                           return seed_of_nodes[seed_incoming_nodes[position]] >= 0;
                                       }),
                        tmp_positions.end());
    if (seed_size < tmp_positions.size()) {
        std::nth_element(tmp_positions.begin(),
                         tmp_positions.begin() + seed_size,
                         tmp_positions.end(),
                         [&](const size_t left_position, const size_t right_position) {
                             return seed_incoming_weights[left_position] > seed_incoming_weights[right_position];
                         });
        tmp_positions.resize(seed_size);
    }

    FastAssertCompare(seed_of_nodes[seed_node_index], <, 0);
    seed_of_nodes[seed_node_index] = int32_t(seed_index);
    connected_nodes[seed_node_index].clear();

    for (auto position : tmp_positions) {
        auto node_index = seed_incoming_nodes[position];
        connected_nodes[node_index].clear();
        seed_of_nodes[node_index] = int32_t(seed_index);
    }

    auto outgoing_nodes = outgoing_weights.get_band_indices(seed_node_index);
    for (size_t node_index : outgoing_nodes) {
        if (seed_of_nodes[node_index] < 0) {
            remove_sorted(connected_nodes[node_index], int32_t(seed_node_index));
        }
    }

    for (auto position : tmp_positions) {
        auto node_index = seed_incoming_nodes[position];
        auto outgoing_nodes = outgoing_weights.get_band_indices(node_index);
        for (size_t other_node_index : outgoing_nodes) {
            if (seed_of_nodes[other_node_index] < 0) {
                remove_sorted(connected_nodes[other_node_index], node_index);
            }
        }
    }
}

static bool
keep_large_candidates(std::vector<size_t>& tmp_candidates, const std::vector<std::vector<int32_t>>& connected_nodes) {
    tmp_candidates.erase(std::remove_if(tmp_candidates.begin(),
                                        tmp_candidates.end(),
                                        [&](size_t candidate_node_index) {
                                            return connected_nodes[candidate_node_index].size() == 0;
                                        }),
                         tmp_candidates.end());
    return tmp_candidates.size() > 0;
}

/// See the Python `metacell.tools.candidates._choose_seeds` function.
void
choose_seeds(const pybind11::array_t<float32_t>& outgoing_weights_data_array,
             const pybind11::array_t<int32_t>& outgoing_weights_indices_array,
             const pybind11::array_t<int32_t>& outgoing_weights_indptr_array,
             const pybind11::array_t<float32_t>& incoming_weights_data_array,
             const pybind11::array_t<int32_t>& incoming_weights_indices_array,
             const pybind11::array_t<int32_t>& incoming_weights_indptr_array,
             const size_t random_seed,
             const size_t max_seeds_count,
             const float32_t min_seed_size_quantile,
             const float32_t max_seed_size_quantile,
             pybind11::array_t<int32_t>& seed_of_nodes_array) {
    WithoutGil without_gil{};
    ArraySlice<int32_t> seed_of_nodes = ArraySlice<int32_t>(seed_of_nodes_array, "seed_of_nodes");
    size_t nodes_count = seed_of_nodes.size();
    FastAssertCompare(nodes_count, >, 0);

    ConstCompressedMatrix<float32_t, int32_t, int32_t>
        outgoing_weights(ConstArraySlice<float32_t>(outgoing_weights_data_array, "outgoing_weights_data"),
                         ConstArraySlice<int32_t>(outgoing_weights_indices_array, "outgoing_weights_indices"),
                         ConstArraySlice<int32_t>(outgoing_weights_indptr_array, "outgoing_weights_indptr"),
                         int32_t(nodes_count),
                         "outgoing_weights");
    FastAssertCompare(outgoing_weights.bands_count(), ==, nodes_count);

    ConstCompressedMatrix<float32_t, int32_t, int32_t>
        incoming_weights(ConstArraySlice<float32_t>(incoming_weights_data_array, "incoming_weights_data"),
                         ConstArraySlice<int32_t>(incoming_weights_indices_array, "incoming_weights_indices"),
                         ConstArraySlice<int32_t>(incoming_weights_indptr_array, "incoming_weights_indptr"),
                         int32_t(nodes_count),
                         "incoming_weights");
    FastAssertCompare(incoming_weights.bands_count(), ==, nodes_count);

    FastAssertCompare(0, <=, min_seed_size_quantile);
    FastAssertCompare(min_seed_size_quantile, <=, max_seed_size_quantile);
    FastAssertCompare(max_seed_size_quantile, <=, 1.0);

    std::vector<std::vector<int32_t>> connected_nodes = collect_connected_nodes(incoming_weights, seed_of_nodes);

    TmpVectorSizeT tmp_candidates_raii;
    auto tmp_candidates = tmp_candidates_raii.vector(nodes_count);
    tmp_candidates.clear();
    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        if (seed_of_nodes[node_index] < 0) {
            tmp_candidates.push_back(node_index);
        }
    }

    std::minstd_rand random(random_seed);
    size_t given_seeds_count = size_t(*std::max_element(seed_of_nodes.begin(), seed_of_nodes.end()) + 1);
    size_t seeds_count = given_seeds_count;

    FastAssertCompare(tmp_candidates.size(), >=, max_seeds_count - given_seeds_count);
    size_t mean_seed_size = size_t(ceil(tmp_candidates.size() / (max_seeds_count - given_seeds_count)));
    FastAssertCompare(mean_seed_size, >=, 1);

    while (seeds_count < max_seeds_count && keep_large_candidates(tmp_candidates, connected_nodes)) {
        size_t seed_node_index =
            choose_seed_node(tmp_candidates, connected_nodes, min_seed_size_quantile, max_seed_size_quantile, random);

        store_seed_node(outgoing_weights,
                        incoming_weights,
                        seeds_count,
                        seed_node_index,
                        tmp_candidates,
                        connected_nodes,
                        seed_of_nodes,
                        mean_seed_size);
        ++seeds_count;
    }

    size_t strong_seeds_count = seeds_count;
    FastAssertCompare(strong_seeds_count, >, given_seeds_count);

    if (seeds_count < max_seeds_count) {
        tmp_candidates.clear();
        for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
            if (seed_of_nodes[node_index] < 0) {
                tmp_candidates.push_back(node_index);
            }
        }
        std::sort(tmp_candidates.begin(),
                  tmp_candidates.end(),
                  [&](const size_t left_node_index, const size_t right_node_index) {
                      const auto left_node_incoming = incoming_weights.get_band_indices(left_node_index);
                      const auto right_node_incoming = incoming_weights.get_band_indices(right_node_index);
                      const auto left_node_outgoing = outgoing_weights.get_band_indices(left_node_index);
                      const auto right_node_outgoing = outgoing_weights.get_band_indices(right_node_index);
                      return (left_node_incoming.size() + 1) * (left_node_outgoing.size() + 1)
                             > (right_node_incoming.size() + 1) * (right_node_outgoing.size() + 1);
                  });

        while (tmp_candidates.size() > 0 && seeds_count < max_seeds_count) {
            auto node_index = tmp_candidates.back();
            tmp_candidates.pop_back();
            seed_of_nodes[node_index] = int32_t(seeds_count);
            ++seeds_count;
        }
    }

    FastAssertCompare(seeds_count, ==, max_seeds_count);
}

void
register_choose_seeds(pybind11::module& module) {
    module.def("choose_seeds", &metacells::choose_seeds, "Choose seed partitions for computing metacells.");
}

}
