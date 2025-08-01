// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <utility>
#include <vector>

#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::ccl::common {

namespace detail {

bool has_wrap_around(tt::tt_fabric::Topology topology) {
    return topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus;
}

std::vector<tt::tt_metal::IDevice*> get_axis_devices(
    const MeshDeviceView& mesh_view, uint32_t axis, uint32_t axis_value) {
    // axis == 1 -> horizontal row (East/West)
    // axis == 0 -> vertical column (North/South)
    if (axis == 1) {
        return mesh_view.get_devices_on_row(axis_value);
    } else if (axis == 0) {
        return mesh_view.get_devices_on_column(axis_value);
    }
    TT_THROW("Axis must be 0 (column) or 1 (row)");
    return {};
}

uint32_t device_index(const std::vector<tt::tt_metal::IDevice*>& devices, const tt::tt_metal::IDevice* device) {
    for (uint32_t i = 0; i < devices.size(); i++) {
        if (devices[i] == device) {
            return i;
        }
    }
    TT_THROW("Device not found in device_index");
    return std::numeric_limits<uint32_t>::max();
}
}  // namespace detail

std::pair<std::vector<tt::tt_metal::IDevice*>, std::array<bool, 4>> get_neighbors(
    const MeshDeviceView& mesh_view,
    const MeshCoordinate& mesh_coordinate,
    const tt::tt_fabric::Topology topology,
    const std::optional<uint32_t> axis) {
    // For readability use symbolic indices instead of raw numbers when accessing the
    // `directions` array `{East, West, North, South}`.
    enum Direction : std::size_t { East = 0, West = 1, North = 2, South = 3 };

    std::vector<tt::tt_metal::IDevice*> neighbors;
    // directions: {East, West, North, South}
    std::array<bool, 4> directions = {false, false, false, false};

    const bool wrap_around_connection = detail::has_wrap_around(topology);
    auto src_device = mesh_view.get_device(mesh_coordinate);

    // Helper that appends neighbours for a single axis
    auto process_axis = [&](uint32_t axis_val) {
        auto axis_devices =
            detail::get_axis_devices(mesh_view, axis_val, axis_val == 1 ? mesh_coordinate[0] : mesh_coordinate[1]);
        uint32_t idx = detail::device_index(axis_devices, src_device);
        uint32_t size = axis_devices.size();
        if (size <= 1) {
            return;  // no neighbours on this axis
        }
        uint32_t next_neighbor_idx = idx + 1;
        uint32_t prev_neighbor_idx = idx - 1;
        uint32_t first_device = 0;
        uint32_t last_device = size - 1;

        auto add_neighbor = [&](Direction dir, uint32_t dev_idx) {
            neighbors.push_back(axis_devices[dev_idx]);
            directions[dir] = true;
        };

        if (axis_val == 1) {
            // For horizontal axis (rows): process East then West
            // Positive direction (East)
            if (next_neighbor_idx < size) {
                log_debug(tt::LogOp, "Adding East neighbor: {}", next_neighbor_idx);
                add_neighbor(Direction::East, next_neighbor_idx);
            } else if (wrap_around_connection) {
                add_neighbor(Direction::East, first_device);
            }

            // Negative direction (West)
            if (idx > 0) {
                log_debug(tt::LogOp, "Adding West neighbor: {}", prev_neighbor_idx);
                add_neighbor(Direction::West, prev_neighbor_idx);
            } else if (wrap_around_connection) {
                add_neighbor(Direction::West, last_device);
            }
        } else {
            // For vertical axis (columns): process North then South to maintain correct order
            // Negative direction (North)
            if (idx > 0) {
                log_debug(tt::LogOp, "Adding North neighbor: {}", prev_neighbor_idx);
                add_neighbor(Direction::North, prev_neighbor_idx);
            } else if (wrap_around_connection) {
                add_neighbor(Direction::North, last_device);
            }

            // Positive direction (South)
            if (next_neighbor_idx < size) {
                log_debug(tt::LogOp, "Adding South neighbor: {}", next_neighbor_idx);
                add_neighbor(Direction::South, next_neighbor_idx);
            } else if (wrap_around_connection) {
                add_neighbor(Direction::South, first_device);
            }
        }
    };

    if (axis.has_value()) {
        process_axis(axis.value());
    } else {
        // When no axis is specified, gather neighbours on both axes
        process_axis(1);  // horizontal (row)
        process_axis(0);  // vertical (column)
    }

    TT_FATAL(neighbors.size() > 0, "No neighbors found");
    TT_FATAL(!(axis.has_value() && neighbors.size() > 2), "Along a single axis, there can only be 2 neighbors");

    if (!axis.has_value()) {
        TT_FATAL(!(wrap_around_connection && neighbors.size() != 4), "Ring/Torus topology must have 4 neighbors");
    }

    return {neighbors, directions};
}

uint32_t get_linearized_index(const ttnn::MeshCoordinate& mesh_coordinate, const ttnn::MeshDeviceView& mesh_view) {
    return mesh_coordinate[0] * mesh_view.num_cols() + mesh_coordinate[1];
}

}  // namespace ttnn::operations::ccl::common
