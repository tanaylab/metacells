#include "metacells/extensions.h"

namespace metacells {

// Score information for one node for one partition.
struct NodeScore {
private:
    float64_t m_total_outgoing_weights;
    float64_t m_total_incoming_weights;
    float64_t m_score;

public:
    NodeScore() : m_total_outgoing_weights(0), m_total_incoming_weights(0), m_score(log2(EPSILON) / 2.0) {}

    void update_outgoing(const int direction, const float64_t edge_weight) {
        m_total_outgoing_weights += direction * edge_weight;
        SlowAssertCompare(m_total_outgoing_weights, >=, -EPSILON);
        m_total_outgoing_weights = std::max(m_total_outgoing_weights, 0.0);
        m_score = NaN;
    }

    void update_incoming(const int direction, const float64_t edge_weight) {
        m_total_incoming_weights += direction * edge_weight;
        SlowAssertCompare(m_total_incoming_weights, >=, -EPSILON);
        m_total_incoming_weights = std::max(m_total_incoming_weights, 0.0);
        m_score = NaN;
    }

    float64_t total_outgoing_weights() const { return m_total_outgoing_weights; }

    float64_t total_incoming_weights() const { return m_total_incoming_weights; }

    float64_t score() const {
        SlowAssertCompare(std::isnan(m_score), ==, false);
        return m_score;
    }

    float64_t rescore() {
        SlowAssertCompare(m_total_outgoing_weights, >=, 0);
        SlowAssertCompare(m_total_outgoing_weights, <=, 1 + EPSILON);
        SlowAssertCompare(m_total_incoming_weights, >=, 0);
        m_score = log2(EPSILON + m_total_outgoing_weights * m_total_incoming_weights) / 2.0;
        return m_score;
    }
};

static std::ostream&
operator<<(std::ostream& os, const NodeScore& node_score) {
    return os << node_score.score() << " total_outgoing_weights: " << node_score.total_outgoing_weights()
              << " total_incoming_weights: " << node_score.total_incoming_weights();
}

static std::vector<size_t>
initial_size_of_partitions(ConstArraySlice<int32_t> partitions_of_nodes) {
    std::vector<size_t> size_of_partitions;

    for (int32_t partition_index : partitions_of_nodes) {
        if (partition_index < 0) {
            continue;
        }
        if (size_t(partition_index) >= size_of_partitions.size()) {
            size_of_partitions.resize(partition_index + 1);
        }
        ++size_of_partitions[partition_index];
    }

    for (size_t partition_index = 0; partition_index < size_of_partitions.size(); ++partition_index) {
        FastAssertCompare(size_of_partitions[partition_index], >, 0);
    }

    return size_of_partitions;
}

static float64_t
initial_incoming_scale(ConstCompressedMatrix<float32_t, int32_t, int32_t> incoming_weights) {
    size_t nodes_count = incoming_weights.bands_count();
    float64_t incoming_scale = 0;

    LOCATED_LOG(false) << " initial_incoming_scale_of_nodes" << std::endl;
    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        float64_t total_incoming_weights = 0;
        const auto& node_incoming_weights = incoming_weights.get_band_data(node_index);
        for (const auto incoming_weight : node_incoming_weights) {
            total_incoming_weights += incoming_weight;
        }
        LOCATED_LOG(false)                                            //
            << " node_index: " << node_index                          //
            << " total_incoming_weights: " << total_incoming_weights  //
            << std::endl;
        FastAssertCompare(total_incoming_weights, >, 0);
        incoming_scale += log2(total_incoming_weights);
    }

    return incoming_scale;
}

static std::vector<std::vector<NodeScore>>
initial_score_of_nodes_of_partitions(ConstCompressedMatrix<float32_t, int32_t, int32_t> outgoing_weights,
                                     ConstCompressedMatrix<float32_t, int32_t, int32_t> incoming_weights,
                                     ConstArraySlice<int32_t> partition_of_nodes,
                                     size_t partitions_count) {
    LOCATED_LOG(false) << " initial_score_of_partitions" << std::endl;
    size_t nodes_count = outgoing_weights.bands_count();

    std::vector<std::vector<NodeScore>> score_of_nodes_of_partitions(partitions_count);
    for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
        score_of_nodes_of_partitions[partition_index].resize(nodes_count);
    }

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        const int partition_index = partition_of_nodes[node_index];
#if ASSERT_LEVEL > 0
        float64_t total_outgoing_weights = 0;
#endif

        const auto& node_outgoing_indices = outgoing_weights.get_band_indices(node_index);
        const auto& node_outgoing_weights = outgoing_weights.get_band_data(node_index);
        for (size_t other_node_position = 0; other_node_position < node_outgoing_indices.size();
             ++other_node_position) {
            const int32_t other_node_index = node_outgoing_indices[other_node_position];
            const float32_t edge_weight = node_outgoing_weights[other_node_position];

            SlowAssertCompare(other_node_index, !=, node_index);
            SlowAssertCompare(edge_weight, >, 0);
            SlowAssertCompare(edge_weight, <, 1 + EPSILON);

#if ASSERT_LEVEL > 0
            total_outgoing_weights += edge_weight;
#endif

            const int other_partition_index = partition_of_nodes[other_node_index];

            LOCATED_LOG(false)                                       //
                << " from node_index: " << node_index                //
                << " from partition_index: " << partition_index      //
                << " to_node_index: " << other_node_index            //
                << " to_partition_index: " << other_partition_index  //
                << " edge weight: " << edge_weight                   //
                << std::endl;

            if (other_partition_index >= 0) {
                score_of_nodes_of_partitions[other_partition_index][node_index].update_outgoing(+1, edge_weight);
                score_of_nodes_of_partitions[other_partition_index][node_index].rescore();
            }

            if (partition_index >= 0) {
                score_of_nodes_of_partitions[partition_index][other_node_index].update_incoming(+1, edge_weight);
                score_of_nodes_of_partitions[partition_index][other_node_index].rescore();
            }
        }

#if ASSERT_LEVEL > 0
        FastAssertCompare(total_outgoing_weights, >, 1 - EPSILON);
        FastAssertCompare(total_outgoing_weights, <, 1 + EPSILON);
#endif
    }

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            if (int32_t(partition_index) == partition_of_nodes[node_index]) {
                LOCATED_LOG(false)                                                              //
                    << " node_index: " << node_index                                            //
                    << " current partition_index: " << partition_index                          //
                    << " score: " << score_of_nodes_of_partitions[partition_index][node_index]  //
                    << std::endl;
            } else {
                LOCATED_LOG(false)                                                              //
                    << " node_index: " << node_index                                            //
                    << " other partition_index: " << partition_index                            //
                    << " score: " << score_of_nodes_of_partitions[partition_index][node_index]  //
                    << std::endl;
            }
        }
    }

    return score_of_nodes_of_partitions;
}

static std::vector<float64_t>
initial_score_of_partitions(size_t nodes_count,
                            ConstArraySlice<int32_t> partitions_of_nodes,
                            const size_t partitions_count,
                            const std::vector<std::vector<NodeScore>>& score_of_nodes_of_partitions) {
    std::vector<float64_t> score_of_partitions(partitions_count, 0);

    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
        const int partition_index = partitions_of_nodes[node_index];
        if (partition_index >= 0) {
            score_of_partitions[partition_index] += score_of_nodes_of_partitions[partition_index][node_index].score();
        }
    }

    return score_of_partitions;
}

#if ASSERT_LEVEL > 0
std::function<void()> g_verify;
#endif

// Optimize the partition of a graph.
struct OptimizePartitions {
    ConstCompressedMatrix<float32_t, int32_t, int32_t> outgoing_weights;
    ConstCompressedMatrix<float32_t, int32_t, int32_t> incoming_weights;
    const size_t nodes_count;
    ArraySlice<int32_t> partition_of_nodes;
    std::vector<float64_t> temperature_of_nodes;
    std::vector<size_t> size_of_partitions;
    const size_t partitions_count;
    float64_t incoming_scale;
    std::vector<std::vector<NodeScore>> score_of_nodes_of_partitions;
    std::vector<float64_t> score_of_partitions;

    OptimizePartitions(const pybind11::array_t<float32_t>& outgoing_weights_array,
                       const pybind11::array_t<int32_t>& outgoing_indices_array,
                       const pybind11::array_t<int32_t>& outgoing_indptr_array,
                       const pybind11::array_t<float32_t>& incoming_weights_array,
                       const pybind11::array_t<int32_t>& incoming_indices_array,
                       const pybind11::array_t<int32_t>& incoming_indptr_array,
                       pybind11::array_t<int32_t>& partition_of_nodes_array)
      : outgoing_weights(ConstCompressedMatrix<float32_t, int32_t, int32_t>(
          ConstArraySlice<float32_t>(outgoing_weights_array, "outgoing_weights_array"),
          ConstArraySlice<int32_t>(outgoing_indices_array, "outgoing_indices_array"),
          ConstArraySlice<int32_t>(outgoing_indptr_array, "outgoing_indptr_array"),
          int32_t(outgoing_indptr_array.size() - 1),
          "outgoing_weights"))
      , incoming_weights(ConstCompressedMatrix<float32_t, int32_t, int32_t>(
            ConstArraySlice<float32_t>(incoming_weights_array, "incoming_weights_array"),
            ConstArraySlice<int32_t>(incoming_indices_array, "incoming_indices_array"),
            ConstArraySlice<int32_t>(incoming_indptr_array, "incoming_indptr_array"),
            int32_t(incoming_indptr_array.size() - 1),
            "incoming_weights"))
      , nodes_count(outgoing_weights.bands_count())
      , partition_of_nodes(partition_of_nodes_array, "partition_of_nodes")
      , temperature_of_nodes(nodes_count, 1.0)
      , size_of_partitions(initial_size_of_partitions(partition_of_nodes))
      , partitions_count(size_of_partitions.size())
      , incoming_scale(initial_incoming_scale(incoming_weights))
      , score_of_nodes_of_partitions(initial_score_of_nodes_of_partitions(outgoing_weights,
                                                                          incoming_weights,
                                                                          partition_of_nodes,
                                                                          partitions_count))
      , score_of_partitions(initial_score_of_partitions(nodes_count,
                                                        partition_of_nodes,
                                                        partitions_count,
                                                        score_of_nodes_of_partitions)) {}

    float64_t score(bool with_orphans = true) const {
        /*
        if (!with_orphans) {
            std::cerr << "node,partition,total_outgoing,total_incoming" << std::endl;
            for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
                for (size_t partition_index = 0; partition_index < partitions_count;
                     ++partition_index) {
                    std::cerr << node_index << ","       //
                              << partition_index << ","  //
                              << score_of_nodes_of_partitions[partition_index][node_index]
                                     .total_outgoing_weights()
                              << ","  //
                              << score_of_nodes_of_partitions[partition_index][node_index]
                                     .total_incoming_weights()  //
                              << std::endl;
                }
            }
        }
        */

        float64_t total_score = nodes_count * log2(float64_t(nodes_count)) - incoming_scale;
        size_t orphans_count = nodes_count;
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            const float64_t score_of_partition = score_of_partitions[partition_index];
            const size_t size_of_partition = size_of_partitions[partition_index];
            total_score += score_of_partition - size_of_partition * log2(float64_t(size_of_partition));
            orphans_count -= size_of_partition;
        }
        if (with_orphans) {
            total_score += orphans_count * NodeScore().score();
            return total_score / nodes_count;
        } else {
            return total_score / (nodes_count - orphans_count);
        }
    }

#if ASSERT_LEVEL > 0
    void verify(const OptimizePartitions& other) {
#    define ASSERT_SAME(CONTEXT, FIELD, EPSILON)                                                                 \
        if (fabs(double(this_##FIELD) - double(other_##FIELD)) > EPSILON) {                                      \
            std::cerr << "OOPS! " << #CONTEXT << ": " << CONTEXT << " actual " << #FIELD << ": " << this_##FIELD \
                      << ": "                                                                                    \
                      << " computed " << #FIELD << ": " << other_##FIELD << ": " << std::endl;                   \
            assert(false);                                                                                       \
        } else

        ConstArraySlice<int32_t> other_partition_of_nodes = other.partition_of_nodes;
        for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
            auto this_partition_index = partition_of_nodes[node_index];
            auto other_partition_index = other_partition_of_nodes[node_index];
            ASSERT_SAME(node_index, partition_index, 0);
        }

        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
                const auto& this_score_of_node = score_of_nodes_of_partitions[partition_index][node_index];
                const auto& other_score_of_node = other.score_of_nodes_of_partitions[partition_index][node_index];

#    define ASSERT_SCORE_FIELD(FIELD)                                                                                \
        if (fabs(this_score_of_node.FIELD() - other_score_of_node.FIELD()) >= EPSILON) {                             \
            std::cerr << "OOPS! partition_index: " << partition_index << " node_index: " << node_index << " actual " \
                      << #FIELD << ": " << this_score_of_node.FIELD() << " computed " << #FIELD << " : "             \
                      << other_score_of_node.FIELD() << std::endl;                                                   \
            assert(false);                                                                                           \
        } else

                ASSERT_SCORE_FIELD(total_outgoing_weights);
                ASSERT_SCORE_FIELD(total_incoming_weights);
                ASSERT_SCORE_FIELD(score);
            }

            auto this_partition_score = score_of_partitions[partition_index];
            auto other_partition_score = other.score_of_partitions[partition_index];
            ASSERT_SAME(partition_index, partition_score, 1e-3);
        }

        auto this_score = score();
        auto other_score = other.score();
        LOCATED_LOG(false) << " score: " << this_score << " ~ " << other_score << std::endl;
        assert(fabs(this_score - other_score) < 1e-3);
    }
#endif

    void optimize(const size_t random_seed,
                  float64_t cooldown_pass,
                  float64_t cooldown_node,
                  int32_t cold_partitions,
                  float64_t cold_temperature) {
        std::minstd_rand random(random_seed);

        TmpVectorSizeT indices_raii;
        auto tmp_indices = indices_raii.vector(nodes_count);
        std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

        std::vector<uint8_t> frozen_nodes(nodes_count, uint8_t(1));
        size_t frozen_count = 0;
        for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
            auto partition_index = partition_of_nodes[node_index];
            if (0 <= partition_index && partition_index < int32_t(cold_partitions)) {
                temperature_of_nodes[node_index] = cold_temperature;
                frozen_nodes[node_index] = 1;
                ++frozen_count;
            } else if (partition_index < -1) {
                frozen_nodes[node_index] = 1;
                ++frozen_count;
            } else if (partition_index >= -1) {
                temperature_of_nodes[node_index] = 1.0;
                frozen_nodes[node_index] = 0;
            }
            LOCATED_LOG(false)                                           //
                << " node: " << node_index                               //
                << " temperature: " << temperature_of_nodes[node_index]  //
                << std::endl;
        }

        TmpVectorSizeT partitions_raii;
        auto tmp_partitions = partitions_raii.vector(partitions_count);

        TmpVectorFloat64 partition_cold_diffs_raii;
        auto tmp_partition_cold_diffs = partition_cold_diffs_raii.vector(partitions_count);

        TmpVectorFloat64 partition_hot_diffs_raii;
        auto tmp_partition_hot_diffs = partition_hot_diffs_raii.vector(partitions_count);

        FastAssertCompare(cooldown_node, >=, 0.0);
        FastAssertCompare(cooldown_node, <=, 1.0);
        if (cooldown_pass != 0.0) {
            FastAssertCompare(cooldown_pass, >, 0.0);
            FastAssertCompare(cooldown_pass, <, 1.0);
        }
        float64_t cooldown_rate = 1.0 - cooldown_pass / nodes_count;
        float64_t temperature = 1.0;

        bool did_improve = true;
        bool did_skip = false;
        size_t total_skipped = 0;
        size_t total_improved = 0;
        size_t total_unimproved = 0;
        while (temperature > 0 || did_improve) {
            LOCATED_LOG(false)                          //
                << " temperature: " << temperature      //
                << " cooldown_rate: " << cooldown_rate  //
                << " score: " << score()                //
                << " did_improve: " << did_improve      //
                << " did_skip: " << did_skip            //
                << std::endl;
            if (did_improve) {
                did_improve = false;
                did_skip = false;
            } else {
                if (did_skip) {
                    did_skip = false;
                    for (size_t node_index = 0; node_index < nodes_count; ++node_index) {
                        if (!frozen_nodes[node_index]) {
                            float64_t node_temperature = temperature_of_nodes[node_index];
                            temperature = std::min(temperature, node_temperature);
                        }
                    }
                } else {
                    temperature = 0;
                }
                LOCATED_LOG(false)                           //
                    << " next_temperature: " << temperature  //
                    << std::endl;
            }

            std::shuffle(tmp_indices.begin(), tmp_indices.end(), random);
            size_t skipped = 0;
            size_t improved = 0;
            size_t unimproved = 0;
            for (size_t node_index : tmp_indices) {
                auto partition_index = partition_of_nodes[node_index];
                if (partition_index < -1) {
                    continue;
                }
                temperature *= cooldown_rate;
                LOCATED_LOG(false)                                           //
                    << " cooldown_rate: " << cooldown_rate                   //
                    << " temperature: " << temperature                       //
                    << " node: " << node_index                               //
                    << " temperature: " << temperature_of_nodes[node_index]  //
                    << std::endl;
                if (temperature_of_nodes[node_index] < temperature) {
                    did_skip = true;
                    ++skipped;
                    LOCATED_LOG(false)              //
                        << " node: " << node_index  //
                        << " skipped"               //
                        << std::endl;
                } else if (improve_node(node_index,
                                        tmp_partitions,
                                        tmp_partition_cold_diffs,
                                        tmp_partition_hot_diffs,
                                        random,
                                        temperature)) {
                    frozen_count -= frozen_nodes[node_index];
                    frozen_nodes[node_index] = uint8_t(0);
                    did_improve = true;
                    ++improved;
                    LOCATED_LOG(false)              //
                        << " node: " << node_index  //
                        << " improved"              //
                        << std::endl;
                } else {
                    frozen_count -= frozen_nodes[node_index];
                    frozen_nodes[node_index] = uint8_t(0);
                    temperature_of_nodes[node_index] = temperature * (1 - cooldown_node);
                    ++unimproved;
                    LOCATED_LOG(false)                                           //
                        << " node: " << node_index                               //
                        << " unimproved"                                         //
                        << " temperature: " << temperature_of_nodes[node_index]  //
                        << std::endl;
                }
            }
            LOCATED_LOG(false)                                      //
                << " improved: " << improved                        //
                << " unimproved: " << unimproved                    //
                << " skipped: " << skipped                          //
                << " total: " << (improved + unimproved + skipped)  //
                << " frozen: " << frozen_count                      //
                << std::endl;
            total_improved += improved;
            total_skipped += skipped;
            total_unimproved += unimproved;
        }

        LOCATED_LOG(false)                                                        //
            << " improved: " << total_improved                                    //
            << " unimproved: " << total_unimproved                                //
            << " skipped: " << total_skipped                                      //
            << " total: " << (total_improved + total_unimproved + total_skipped)  //
            << " frozen: " << frozen_count                                        //
            << std::endl;
        LOCATED_LOG(false)                          //
            << " temperature: " << temperature      //
            << " cooldown_rate: " << cooldown_rate  //
            << " score: " << score()                //
            << std::endl;
    }

    bool improve_node(size_t node_index,
                      std::vector<size_t>& tmp_partitions,
                      std::vector<float64_t>& tmp_partition_cold_diffs,
                      std::vector<float64_t>& tmp_partition_hot_diffs,
                      std::minstd_rand& random,
                      const float64_t temperature) {
        const auto current_partition_index = partition_of_nodes[node_index];
        LOCATED_LOG(false)                                                                                   //
            << " node: " << node_index                                                                       //
            << " current_partition_index: " << current_partition_index                                       //
            << " size: " << (current_partition_index < 0 ? 0 : size_of_partitions[current_partition_index])  //
            << std::endl;

        if (current_partition_index >= 0 && size_of_partitions[current_partition_index] < 2) {
            return false;
        }

        collect_initial_partition_diffs(node_index,
                                        current_partition_index,
                                        tmp_partition_cold_diffs,
                                        tmp_partition_hot_diffs);

        collect_cold_partition_diffs(node_index, current_partition_index, tmp_partition_cold_diffs);

        const float64_t current_cold_diff = collect_candidate_partitions(current_partition_index,
                                                                         tmp_partition_cold_diffs,
                                                                         tmp_partition_hot_diffs,
                                                                         temperature,
                                                                         tmp_partitions);

        const int32_t chosen_partition_index = choose_target_partition(current_partition_index, random, tmp_partitions);
        if (chosen_partition_index < 0) {
            return false;
        }

        const float64_t chosen_cold_diff = tmp_partition_cold_diffs[chosen_partition_index];
        LOCATED_LOG(false)                                //
            << " chosen_cold_diff: " << chosen_cold_diff  //
            << std::endl;

        update_scores_of_nodes(node_index, current_partition_index, chosen_partition_index);

        update_partitions_of_nodes(node_index, current_partition_index, chosen_partition_index);

        update_sizes_of_partitions(current_partition_index, chosen_partition_index);

        update_scores_of_partition(current_partition_index,
                                   current_cold_diff,
                                   chosen_partition_index,
                                   chosen_cold_diff);

#if ASSERT_LEVEL > 0
        if (g_verify) {
            g_verify();
        }
#endif

        return true;
    }

    void collect_initial_partition_diffs(const size_t node_index,
                                         const int32_t current_partition_index,
                                         std::vector<float64_t>& tmp_partition_cold_diffs,
                                         std::vector<float64_t>& tmp_partition_hot_diffs) {
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            const int direction = 1 - 2 * (int32_t(partition_index) == current_partition_index);
            const auto score = direction * score_of_nodes_of_partitions[partition_index][node_index].score();
            tmp_partition_cold_diffs[partition_index] = score;
            tmp_partition_hot_diffs[partition_index] = score;
            LOCATED_LOG(false)                                                  //
                << " node_index: " << node_index                                //
                << " partition_index: " << partition_index                      //
                << " cold_diff: " << tmp_partition_cold_diffs[partition_index]  //
                << " hot_diff: " << tmp_partition_hot_diffs[partition_index]    //
                << std::endl;
        }
    }

    void collect_cold_partition_diffs(const size_t node_index,
                                      const int32_t current_partition_index,
                                      std::vector<float64_t>& tmp_partition_cold_diffs) {
        const auto& node_outgoing_indices = outgoing_weights.get_band_indices(node_index);
        const auto& node_incoming_indices = incoming_weights.get_band_indices(node_index);

        const auto& node_outgoing_weights = outgoing_weights.get_band_data(node_index);
        const auto& node_incoming_weights = incoming_weights.get_band_data(node_index);

        size_t outgoing_count = node_outgoing_indices.size();
        size_t incoming_count = node_incoming_indices.size();

        FastAssertCompare(outgoing_count, >, 0);
        FastAssertCompare(incoming_count, >, 0);

        size_t outgoing_position = 0;
        size_t incoming_position = 0;

        auto outgoing_node_index = node_outgoing_indices[outgoing_position];
        auto incoming_node_index = node_incoming_indices[incoming_position];

        auto outgoing_edge_weight = node_outgoing_weights[outgoing_position];
        auto incoming_edge_weight = node_incoming_weights[incoming_position];

        while (outgoing_position < outgoing_count || incoming_position < incoming_count) {
            const auto other_node_index = std::min(outgoing_node_index, incoming_node_index);
            const auto other_partition_index = partition_of_nodes[other_node_index];

            const int is_outgoing = int(outgoing_node_index == other_node_index);
            const int is_incoming = int(incoming_node_index == other_node_index);

            LOCATED_LOG(false)                                          //
                << " consider other_node " << other_node_index          //
                << " other_partition_index: " << other_partition_index  //
                << std::endl;

            if (other_partition_index >= 0) {
                NodeScore other_score = score_of_nodes_of_partitions[other_partition_index][other_node_index];
                const float64_t old_score = other_score.score();

                LOCATED_LOG(false)                                          //
                    << " other_node_index: " << other_node_index            //
                    << " other_partition_index: " << other_partition_index  //
                    << " old score: " << other_score                        //
                    << std::endl;

                const int direction = 1 - 2 * (other_partition_index == current_partition_index);

                other_score.update_incoming(direction * is_outgoing, outgoing_edge_weight);
                LOCATED_LOG(false && is_outgoing)                         //
                    << " other_node_index: " << other_node_index          //
                    << " direction: " << direction                        //
                    << " outgoing_edge_weight: " << outgoing_edge_weight  //
                    << std::endl;

                other_score.update_outgoing(direction * is_incoming, incoming_edge_weight);
                LOCATED_LOG(false && is_incoming)                         //
                    << " other_node_index: " << other_node_index          //
                    << " direction: " << direction                        //
                    << " incoming_edge_weight: " << incoming_edge_weight  //
                    << std::endl;

                const float64_t new_score = other_score.rescore();
                LOCATED_LOG(false)                                          //
                    << " other_node_index: " << other_node_index            //
                    << " other_partition_index: " << other_partition_index  //
                    << " new score: " << other_score                        //
                    << std::endl;

                tmp_partition_cold_diffs[other_partition_index] += new_score - old_score;
            }

            outgoing_position += is_outgoing;
            incoming_position += is_incoming;

            if (outgoing_position < outgoing_count) {
                outgoing_node_index = node_outgoing_indices[outgoing_position];
                outgoing_edge_weight = node_outgoing_weights[outgoing_position];
            } else {
                outgoing_node_index = int32_t(nodes_count);
                outgoing_edge_weight = 0;
            }

            if (incoming_position < incoming_count) {
                incoming_node_index = node_incoming_indices[incoming_position];
                incoming_edge_weight = node_incoming_weights[incoming_position];
            } else {
                incoming_node_index = int32_t(nodes_count);
                incoming_edge_weight = 0;
            }
        }
    }

    float64_t collect_candidate_partitions(const int32_t current_partition_index,
                                           const std::vector<float64_t>& tmp_partition_cold_diffs,
                                           const std::vector<float64_t>& tmp_partition_hot_diffs,
                                           const float64_t temperature,
                                           std::vector<size_t>& tmp_partitions) {
        float64_t current_hot_diff = 0;
        float64_t current_cold_diff = 0;
        float64_t current_adjusted_cold_diff = 0;

        if (current_partition_index >= 0) {
            const float64_t hot_diff = tmp_partition_hot_diffs[current_partition_index];
            const size_t old_size = size_of_partitions[current_partition_index];
            const size_t new_size = old_size - 1;
            const float64_t old_score = score_of_partitions[current_partition_index];
            const float64_t old_adjusted_score = old_score - old_size * log2(float64_t(old_size));
            const float64_t cold_diff = tmp_partition_cold_diffs[current_partition_index];
            const float64_t new_score = old_score + cold_diff;
            const float64_t new_adjusted_score = new_score - new_size * log2(float64_t(new_size));
            const float64_t adjusted_cold_diff = new_adjusted_score - old_adjusted_score;
            LOCATED_LOG(false)                                              //
                << " current_partition_index: " << current_partition_index  //
                << " hot_diff: " << hot_diff                                //
                << " old_score: " << old_score                              //
                << " new_score: " << new_score                              //
                << " cold_diff: " << cold_diff                              //
                << " old_size: " << old_size                                //
                << " new_size: " << new_size                                //
                << " old_adjusted_score: " << old_adjusted_score            //
                << " new_adjusted_score: " << new_adjusted_score            //
                << " adjusted_cold_diff: " << adjusted_cold_diff            //
                << std::endl;
            current_cold_diff = cold_diff;
            current_hot_diff = hot_diff;
            current_adjusted_cold_diff = adjusted_cold_diff;
        }

        tmp_partitions.clear();
        for (size_t partition_index = 0; partition_index < partitions_count; ++partition_index) {
            if (int32_t(partition_index) == current_partition_index) {
                continue;
            }
            const float64_t hot_diff = tmp_partition_hot_diffs[partition_index];
            const size_t old_size = size_of_partitions[partition_index];
            const size_t new_size = old_size + 1;
            const float64_t old_score = score_of_partitions[partition_index];
            const float64_t cold_diff = tmp_partition_cold_diffs[partition_index];
            const float64_t new_score = old_score + cold_diff;
            const float64_t old_adjusted_score = old_score - old_size * log2(float64_t(old_size));
            const float64_t new_adjusted_score = new_score - new_size * log2(float64_t(new_size));
            const float64_t adjusted_cold_diff = new_adjusted_score - old_adjusted_score;
            const float64_t total_diff = (current_hot_diff + hot_diff) * temperature
                                         + (current_adjusted_cold_diff + adjusted_cold_diff) * (1.0 - temperature);
            LOCATED_LOG(false)                                    //
                << " partition_index: " << partition_index        //
                << " old_score: " << old_score                    //
                << " new_score: " << new_score                    //
                << " hot_diff: " << hot_diff                      //
                << " cold_diff: " << cold_diff                    //
                << " old_size: " << old_size                      //
                << " new_size: " << new_size                      //
                << " old_adjusted_score: " << old_adjusted_score  //
                << " new_adjusted_score: " << new_adjusted_score  //
                << " adjusted_cold_diff: " << adjusted_cold_diff  //
                << " total_diff: " << total_diff                  //
                << std::endl;
            if (total_diff > EPSILON) {
                tmp_partitions.push_back(partition_index);
            }
        }

        return current_cold_diff;
    }

    int32_t choose_target_partition(const int32_t current_partition_index,
                                    std::minstd_rand& random,
                                    const std::vector<size_t>& tmp_partitions) {
        int32_t chosen_partition_index = -1;
        if (tmp_partitions.size() == 0) {
            if (current_partition_index >= 0) {
                return -1;
            }
            chosen_partition_index = random() % partitions_count;
        } else {
            chosen_partition_index = int32_t(tmp_partitions[random() % tmp_partitions.size()]);
        }

        LOCATED_LOG(false)                                            //
            << " chosen_partition_index: " << chosen_partition_index  //
            << std::endl;

        return chosen_partition_index;
    }

    void update_scores_of_nodes(const size_t node_index,
                                const int32_t from_partition_index,
                                const int32_t to_partition_index) {
        auto score_of_nodes_of_from_partition =
            from_partition_index < 0 ? nullptr : &score_of_nodes_of_partitions[from_partition_index];
        auto& score_of_nodes_of_to_partition = score_of_nodes_of_partitions[to_partition_index];

        const auto& node_outgoing_indices = outgoing_weights.get_band_indices(node_index);
        const auto& node_incoming_indices = incoming_weights.get_band_indices(node_index);

        const auto& node_outgoing_weights = outgoing_weights.get_band_data(node_index);
        const auto& node_incoming_weights = incoming_weights.get_band_data(node_index);

        size_t outgoing_count = node_outgoing_indices.size();
        size_t incoming_count = node_incoming_indices.size();

        FastAssertCompare(outgoing_count, >, 0);
        FastAssertCompare(incoming_count, >, 0);

        size_t outgoing_position = 0;
        size_t incoming_position = 0;

        auto outgoing_node_index = node_outgoing_indices[outgoing_position];
        auto incoming_node_index = node_incoming_indices[incoming_position];

        auto outgoing_edge_weight = node_outgoing_weights[outgoing_position];
        auto incoming_edge_weight = node_incoming_weights[incoming_position];

        while (outgoing_position < outgoing_count || incoming_position < incoming_count) {
            const auto other_node_index = std::min(outgoing_node_index, incoming_node_index);

            const int is_outgoing = int(outgoing_node_index == other_node_index);
            const int is_incoming = int(incoming_node_index == other_node_index);

            if (score_of_nodes_of_from_partition) {
                auto& other_node_from_score = (*score_of_nodes_of_from_partition)[other_node_index];
                LOCATED_LOG(false)                                        //
                    << " other_node_index: " << other_node_index          //
                    << " from_partition_index: " << from_partition_index  //
                    << " old score: " << other_node_from_score            //
                    << std::endl;

                other_node_from_score.update_incoming(-is_outgoing, outgoing_edge_weight);
                other_node_from_score.update_outgoing(-is_incoming, incoming_edge_weight);
                other_node_from_score.rescore();

                LOCATED_LOG(false)                                        //
                    << " other_node_index: " << other_node_index          //
                    << " from_partition_index: " << from_partition_index  //
                    << " new score: " << other_node_from_score            //
                    << std::endl;
            }

            auto& other_node_to_score = score_of_nodes_of_to_partition[other_node_index];
            LOCATED_LOG(false)                                    //
                << " other_node_index: " << other_node_index      //
                << " to_partition_index: " << to_partition_index  //
                << " old score: " << other_node_to_score          //
                << std::endl;

            other_node_to_score.update_incoming(+is_outgoing, outgoing_edge_weight);
            other_node_to_score.update_outgoing(+is_incoming, incoming_edge_weight);
            other_node_to_score.rescore();

            LOCATED_LOG(false)                                    //
                << " other_node_index: " << other_node_index      //
                << " to_partition_index: " << to_partition_index  //
                << " new score: " << other_node_to_score          //
                << std::endl;

            outgoing_position += is_outgoing;
            incoming_position += is_incoming;

            if (outgoing_position < outgoing_count) {
                outgoing_node_index = node_outgoing_indices[outgoing_position];
                outgoing_edge_weight = node_outgoing_weights[outgoing_position];
            } else {
                outgoing_node_index = int32_t(nodes_count);
                outgoing_edge_weight = 0;
            }

            if (incoming_position < incoming_count) {
                incoming_node_index = node_incoming_indices[incoming_position];
                incoming_edge_weight = node_incoming_weights[incoming_position];
            } else {
                incoming_node_index = int32_t(nodes_count);
                incoming_edge_weight = 0;
            }
        }
    }

    void update_partitions_of_nodes(const size_t node_index,
                                    const int32_t from_partition_index,
                                    const int32_t to_partition_index) {
        SlowAssertCompare(partition_of_nodes[node_index], ==, from_partition_index);
        partition_of_nodes[node_index] = to_partition_index;
        LOCATED_LOG(false)                                        //
            << " set node_index: " << node_index                  //
            << " from_partition_index: " << from_partition_index  //
            << " to_partition_index: " << to_partition_index      //
            << std::endl;
    }

    void update_sizes_of_partitions(const int32_t from_partition_index, const int32_t to_partition_index) {
        if (from_partition_index >= 0) {
            SlowAssertCompare(size_of_partitions[from_partition_index], >, 1);
            size_of_partitions[from_partition_index] -= 1;
        }

        ++size_of_partitions[to_partition_index];
    }

    void update_scores_of_partition(size_t current_partition_index,
                                    float64_t current_cold_diff,
                                    size_t chosen_partition_index,
                                    float64_t chosen_cold_diff) {
        if (current_partition_index >= 0) {
            LOCATED_LOG(false)                                                     //
                << " from_partition_index: " << current_partition_index            //
                << " old score: " << score_of_partitions[current_partition_index]  //
                << " from_cold_diff: " << current_cold_diff                        //
                << std::endl;
            score_of_partitions[current_partition_index] += current_cold_diff;
            LOCATED_LOG(false)                                                     //
                << " from_partition_index: " << current_partition_index            //
                << " new score: " << score_of_partitions[current_partition_index]  //
                << std::endl;
        }

        LOCATED_LOG(false)                                                    //
            << " to_partition_index: " << chosen_partition_index              //
            << " old score: " << score_of_partitions[chosen_partition_index]  //
            << " to_cold_diff: " << chosen_cold_diff                          //
            << std::endl;
        score_of_partitions[chosen_partition_index] += chosen_cold_diff;
        LOCATED_LOG(false)                                                    //
            << " to_partition_index: " << chosen_partition_index              //
            << " new score: " << score_of_partitions[chosen_partition_index]  //
            << std::endl;
    }
};

static float64_t
optimize_partitions(const pybind11::array_t<float32_t>& outgoing_weights_array,
                    const pybind11::array_t<int32_t>& outgoing_indices_array,
                    const pybind11::array_t<int32_t>& outgoing_indptr_array,
                    const pybind11::array_t<float32_t>& incoming_weights_array,
                    const pybind11::array_t<int32_t>& incoming_indices_array,
                    const pybind11::array_t<int32_t>& incoming_indptr_array,
                    const unsigned int random_seed,
                    float64_t cooldown_pass,
                    float64_t cooldown_node,
                    pybind11::array_t<int32_t>& partition_of_nodes_array,
                    int32_t cold_partitions,
                    float64_t cold_temperature) {
    WithoutGil without_gil{};
    OptimizePartitions optimizer(outgoing_weights_array,
                                 outgoing_indices_array,
                                 outgoing_indptr_array,
                                 incoming_weights_array,
                                 incoming_indices_array,
                                 incoming_indptr_array,
                                 partition_of_nodes_array);

#if ASSERT_LEVEL > 1
    g_verify = [&]() {
        LOCATED_LOG(false) << " VERIFY" << std::endl;
        OptimizePartitions verifier(outgoing_weights_array,
                                    outgoing_indices_array,
                                    outgoing_indptr_array,
                                    incoming_weights_array,
                                    incoming_indices_array,
                                    incoming_indptr_array,
                                    partition_of_nodes_array);
        LOCATED_LOG(false) << " COMPARE" << std::endl;
        verifier.verify(optimizer);
        LOCATED_LOG(false) << " VERIFIED" << std::endl;
    };
#else
    g_verify = nullptr;
#endif

    optimizer.optimize(random_seed, cooldown_pass, cooldown_node, cold_partitions, cold_temperature);

    float64_t score = optimizer.score();
    return score;
}

static float64_t
score_partitions(const pybind11::array_t<float32_t>& outgoing_weights_array,
                 const pybind11::array_t<int32_t>& outgoing_indices_array,
                 const pybind11::array_t<int32_t>& outgoing_indptr_array,
                 const pybind11::array_t<float32_t>& incoming_weights_array,
                 const pybind11::array_t<int32_t>& incoming_indices_array,
                 const pybind11::array_t<int32_t>& incoming_indptr_array,
                 pybind11::array_t<int32_t>& partition_of_nodes_array,
                 bool with_orphans) {
    WithoutGil without_gil{};
    OptimizePartitions optimizer(outgoing_weights_array,
                                 outgoing_indices_array,
                                 outgoing_indptr_array,
                                 incoming_weights_array,
                                 incoming_indices_array,
                                 incoming_indptr_array,
                                 partition_of_nodes_array);
    return optimizer.score(with_orphans);
}

void
register_partitions(pybind11::module& module) {
    module.def("optimize_partitions",
               &metacells::optimize_partitions,
               "Optimize the partition for computing metacells.");
    module.def("score_partitions", &metacells::score_partitions, "Compute the quality score for metacells.");
}

}
