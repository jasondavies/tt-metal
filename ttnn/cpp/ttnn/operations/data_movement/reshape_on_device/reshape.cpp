// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "device/reshape_op.hpp"

#include "ttnn/operations/experimental/reshape/view.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

static Tensor manual_insertion(
    const Tensor& input_tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    tt::tt_metal::distributed::MeshDevice* device,
    const MemoryConfig& output_mem_config) {
    TT_ASSERT(input_tensor.layout() == Layout::ROW_MAJOR);
    TT_ASSERT(
        logical_shape.volume() == input_tensor.logical_volume(),
        "Required shape volume ({}) must match old shape volume ({})",
        logical_shape.volume(),
        input_tensor.logical_volume());
    auto cpu_tensor = input_tensor.cpu();
    auto output =
        Tensor(
            cpu_tensor.storage(),
            TensorSpec(
                logical_shape,
                TensorLayout::fromPaddedShape(
                    DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}, logical_shape, padded_shape)),
            cpu_tensor.distributed_tensor_config(),
            cpu_tensor.tensor_topology())
            .to_layout(Layout::ROW_MAJOR);
    if (device != nullptr) {
        output = output.to_device(device, output_mem_config);
    }
    return output;
}
}  // namespace detail

ttnn::Tensor ReshapeOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_output_shape,
    const ttnn::Shape& padded_output_shape,
    const std::optional<MemoryConfig>& memory_config_arg) {
    using namespace tt::constants;
    auto output_mem_config = memory_config_arg.value_or(input_tensor.memory_config());
    // No-op (Will do a tensor copy)
    if (((input_tensor.layout() == Layout::TILE or input_tensor.layout() == Layout::ROW_MAJOR) &&
         padded_output_shape[3] == input_tensor.padded_shape()[3])) {
        // Don't need to do a check here to see the H and W both divisible by 32
        // since handled within the tensor reshape method
        return ttnn::experimental::view(input_tensor, logical_output_shape, padded_output_shape);
    }
    if (input_tensor.padded_shape() == padded_output_shape) {
        return ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(
            input_tensor, output_mem_config);
    }
    uint32_t ROW_MAJOR_WIDTH = 8;
    if (input_tensor.layout() == Layout::ROW_MAJOR &&
        (input_tensor.padded_shape()[3] % ROW_MAJOR_WIDTH != 0 || padded_output_shape[3] % ROW_MAJOR_WIDTH != 0) &&
        ((padded_output_shape.volume() / padded_output_shape[-1]) % TILE_HEIGHT != 0 ||
         padded_output_shape[-1] % TILE_WIDTH != 0 || input_tensor.padded_shape()[-1] % TILE_WIDTH != 0 ||
         (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) % TILE_HEIGHT != 0)) {
        TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "Error");

        return detail::manual_insertion(
            (tt::tt_metal::Tensor)input_tensor,
            logical_output_shape,
            padded_output_shape,
            input_tensor.device(),
            output_mem_config);
    }
    return tt::tt_metal::operation::run(
               ReshapeDeviceOperation{logical_output_shape, padded_output_shape, output_mem_config}, {input_tensor})
        .at(0);
}

ttnn::Tensor ReshapeOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_output_shape,
    const std::optional<MemoryConfig>& memory_config_arg) {
    return invoke(queue_id, input_tensor, logical_output_shape, logical_output_shape, memory_config_arg);
}

ttnn::Tensor ReshapeOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config_arg) {
    return invoke(queue_id, input_tensor, infer_dims_for_reshape(input_tensor, shape_vector), memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
