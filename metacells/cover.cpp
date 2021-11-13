#include "metacells/extensions.h"

namespace metacells {

static float64_t
cover_diameter(size_t points_count, float64_t area, float64_t cover_fraction) {
    float64_t point_area = area * cover_fraction / points_count;
    return sqrt(point_area) * 4 / M_PI;
}

static const size_t DELTAS_COUNT = 6;

static const ssize_t DELTAS[2][DELTAS_COUNT][2] = {
    { { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 0 } },
    { { -1, -1 }, { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 1 }, { -1, 0 } }
};

static size_t
distance(size_t left_x_index, size_t left_y_index, size_t right_x_index, size_t right_y_index) {
    ssize_t x_distance = ssize_t(left_x_index) - ssize_t(right_x_index);
    ssize_t y_distance = ssize_t(left_y_index) - ssize_t(right_y_index);
    return size_t(x_distance * x_distance + y_distance * y_distance);
}

template<typename D>
static void
cover_coordinates(const pybind11::array_t<D>& raw_x_coordinates_array,
                  const pybind11::array_t<D>& raw_y_coordinates_array,
                  pybind11::array_t<D>& spaced_x_coordinates_array,
                  pybind11::array_t<D>& spaced_y_coordinates_array,
                  const float64_t cover_fraction,
                  const float64_t noise_fraction,
                  const size_t random_seed) {
    WithoutGil without_gil{};
    FastAssertCompare(cover_fraction, >, 0);
    FastAssertCompare(cover_fraction, <, 1);
    FastAssertCompare(noise_fraction, >=, 0);

    ConstArraySlice<D> raw_x_coordinates(raw_x_coordinates_array, "raw_x_coordinates");
    ConstArraySlice<D> raw_y_coordinates(raw_y_coordinates_array, "raw_y_coordinates");
    ArraySlice<D> spaced_x_coordinates(spaced_x_coordinates_array, "spaced_x_coordinates");
    ArraySlice<D> spaced_y_coordinates(spaced_y_coordinates_array, "spaced_y_coordinates");

    size_t points_count = raw_x_coordinates.size();
    FastAssertCompare(points_count, >, 2);
    FastAssertCompare(raw_y_coordinates.size(), ==, points_count);
    FastAssertCompare(spaced_x_coordinates.size(), ==, points_count);
    FastAssertCompare(spaced_y_coordinates.size(), ==, points_count);

    std::minstd_rand random(random_seed);
    std::normal_distribution<float64_t> noise(0.0, noise_fraction);

    const auto x_min = *std::min_element(raw_x_coordinates.begin(), raw_x_coordinates.end());
    const auto y_min = *std::min_element(raw_y_coordinates.begin(), raw_y_coordinates.end());
    const auto x_max = *std::max_element(raw_x_coordinates.begin(), raw_x_coordinates.end());
    const auto y_max = *std::max_element(raw_y_coordinates.begin(), raw_y_coordinates.end());
    const auto x_size = x_max - x_min;
    const auto y_size = y_max - y_min;
    FastAssertCompare(x_size, >, 0);
    FastAssertCompare(y_size, >, 0);

    const auto point_diameter = cover_diameter(points_count, float64_t(x_size) * float64_t(y_size), cover_fraction);

    const auto x_step = point_diameter;
    const auto y_step = point_diameter * sqrt(3.0) / 2.0;

    const size_t x_layout_grid_size = 2 + size_t((x_max - x_min) / x_step);
    const size_t y_layout_grid_size = 2 + size_t((y_max - y_min) / y_step);

    std::vector<std::vector<ssize_t>> point_index_of_location;
    for (size_t x_index = 0; x_index < x_layout_grid_size; ++x_index) {
        point_index_of_location.emplace_back(y_layout_grid_size, -1);
    }

    std::vector<std::array<size_t, 2>> preferred_location_of_points(points_count);
    std::vector<std::array<size_t, 2>> location_of_points(points_count);

    std::vector<size_t> delta_indices(DELTAS_COUNT);
    std::iota(delta_indices.begin(), delta_indices.end(), 0);

    for (size_t each_point_index = 0; each_point_index < points_count; ++each_point_index) {
        ssize_t point_index = ssize_t(each_point_index);
        size_t y_index = size_t(round((raw_y_coordinates[point_index] - y_min) / y_step));
        size_t x_index;
        if (y_index % 2 == 0) {
            x_index = size_t(round((raw_x_coordinates[point_index] - x_min) / x_step));
        } else {
            x_index = size_t(round((raw_x_coordinates[point_index] - x_min) / x_step + 0.5));
        }
        preferred_location_of_points[point_index] = { x_index, y_index };

        while (point_index >= 0) {
            ssize_t other_point_index = point_index_of_location[x_index][y_index];
            if (other_point_index >= 0) {
                std::shuffle(delta_indices.begin(), delta_indices.end(), random);
                for (auto delta_index : delta_indices) {
                    auto delta_x = DELTAS[y_index % 2][delta_index][0];
                    auto delta_y = DELTAS[y_index % 2][delta_index][1];

                    auto other_x_index = x_index + delta_x;
                    auto other_y_index = y_index + delta_y;
                    if (other_x_index < 0 || x_layout_grid_size <= other_x_index || other_y_index < 0
                        || y_layout_grid_size <= other_y_index) {
                        continue;
                    }

                    other_point_index = point_index_of_location[other_x_index][other_y_index];
                    if (other_point_index < 0) {
                        x_index = other_x_index;
                        y_index = other_y_index;
                        break;
                    }
                }
            }

            if (other_point_index >= 0) {
                std::shuffle(delta_indices.begin(), delta_indices.end(), random);
                for (auto delta_index : delta_indices) {
                    auto delta_x = DELTAS[y_index % 2][delta_index][0];
                    auto delta_y = DELTAS[y_index % 2][delta_index][1];

                    auto other_x_index = x_index + delta_x;
                    auto other_y_index = y_index + delta_y;
                    if (other_x_index < 0 || x_layout_grid_size <= other_x_index || other_y_index < 0
                        || y_layout_grid_size <= other_y_index) {
                        continue;
                    }

                    other_point_index = point_index_of_location[other_x_index][other_y_index];
                    x_index = other_x_index;
                    y_index = other_y_index;
                    break;
                }
            }

            point_index_of_location[x_index][y_index] = point_index;
            location_of_points[point_index] = { x_index, y_index };
            point_index = other_point_index;
        }
    }

    auto verify_indices = [&]() {
        for (size_t x_index = 0; x_index < x_layout_grid_size; ++x_index) {
            for (size_t y_index = 0; y_index < y_layout_grid_size; ++y_index) {
                const auto point_index = point_index_of_location[x_index][y_index];
                if (point_index >= 0) {
                    const auto point_x_index = location_of_points[point_index][0];
                    const auto point_y_index = location_of_points[point_index][1];
                    FastAssertCompare(point_x_index, ==, x_index);
                    FastAssertCompare(point_y_index, ==, y_index);
                }
            }
        }

        for (size_t point_index = 0; point_index < points_count; ++point_index) {
            const auto x_index = location_of_points[point_index][0];
            const auto y_index = location_of_points[point_index][1];
            const auto location_point_index = point_index_of_location[x_index][y_index];
            FastAssertCompare(location_point_index, ==, point_index);
        }
    };

    verify_indices();

    for (size_t point_index = 0; point_index < points_count; ++point_index) {
        auto x_index = location_of_points[point_index][0];
        auto y_index = location_of_points[point_index][1];

        auto preferred_x_index = preferred_location_of_points[point_index][0];
        auto preferred_y_index = preferred_location_of_points[point_index][1];

        bool did_move = true;
        while (did_move) {
            did_move = false;
            auto current_distance = distance(x_index, y_index, preferred_x_index, preferred_y_index);

            std::shuffle(delta_indices.begin(), delta_indices.end(), random);
            for (auto delta_index : delta_indices) {
                auto delta_x = DELTAS[y_index % 2][delta_index][0];
                auto delta_y = DELTAS[y_index % 2][delta_index][1];

                auto near_x_index = x_index + delta_x;
                auto near_y_index = y_index + delta_y;
                if (near_x_index < 0 || x_layout_grid_size <= near_x_index || near_y_index < 0
                    || y_layout_grid_size <= near_y_index) {
                    continue;
                }

                auto near_distance = distance(near_x_index, near_y_index, preferred_x_index, preferred_y_index);
                if (near_distance > current_distance) {
                    continue;
                }

                auto other_point_index = point_index_of_location[near_x_index][near_y_index];
                if (other_point_index < 0) {
                    continue;
                }

                auto other_preferred_x_index = preferred_location_of_points[other_point_index][0];
                auto other_preferred_y_index = preferred_location_of_points[other_point_index][1];

                auto other_current_distance =
                    distance(x_index, y_index, other_preferred_x_index, other_preferred_y_index);
                auto other_near_distance =
                    distance(near_x_index, near_y_index, other_preferred_x_index, other_preferred_y_index);
                if (other_current_distance > other_near_distance
                    || (near_distance == current_distance && other_current_distance == other_near_distance)) {
                    continue;
                }

                point_index_of_location[x_index][y_index] = other_point_index;
                location_of_points[other_point_index] = { x_index, y_index };

                point_index_of_location[near_x_index][near_y_index] = point_index;
                location_of_points[point_index] = { near_x_index, near_y_index };

                x_index = near_x_index;
                y_index = near_y_index;
                did_move = true;
            }
        }
    }

    verify_indices();

    for (size_t point_index = 0; point_index < points_count; ++point_index) {
        const auto x_index = location_of_points[point_index][0];
        const auto y_index = location_of_points[point_index][1];
        spaced_y_coordinates[point_index] = D((y_index + noise(random)) * y_step + y_min);
        if (y_index % 2 == 0) {
            spaced_x_coordinates[point_index] = D((x_index + noise(random)) * x_step + x_min);
        } else {
            spaced_x_coordinates[point_index] = D((x_index + noise(random) - 0.5) * x_step + x_min);
        }
    }
}

void
register_cover(pybind11::module& module) {
    module.def("cover_diameter", &metacells::cover_diameter, "The diameter for points to achieve plot area coverage.");

#define REGISTER_D(D) \
    module.def("cover_coordinates_" #D, &metacells::cover_coordinates<D>, "Move points to achieve plot area coverage.");

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
