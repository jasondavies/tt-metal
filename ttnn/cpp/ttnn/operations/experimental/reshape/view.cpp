// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tracy/Tracy.hpp>

namespace ttnn::operations::experimental::reshape {

static MemoryConfig infer_output_memory_config(
    const MemoryConfig& input_memory_config, const ttnn::Shape& output_padded_shape) {
    if (input_memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto shard_spec = input_memory_config.shard_spec().value();
        shard_spec.shape[1] = output_padded_shape[-1];  // update output shard to match new shard width
        return MemoryConfig{input_memory_config.memory_layout(), input_memory_config.buffer_type(), shard_spec};
    } else {
        return input_memory_config;
    }
}

Tensor tensor_reshape(
    const Tensor& input_tensor, const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) {
    ZoneScoped;
    tt::tt_metal::GraphTracker::instance().track_function_start(
        "Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);

    const auto output_memory_config = infer_output_memory_config(input_tensor.memory_config(), new_padded_shape);
    auto new_spec = ttnn::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    // TODO (#25340): Review tensor topology logic for reshape
    auto output = std::visit(
        [&input_tensor, &new_spec, &new_logical_shape, &new_padded_shape](auto&& storage) -> Tensor {
            using T = std::decay_t<decltype(storage)>;
            const auto& tensor = input_tensor;

            if constexpr (std::is_same_v<T, tt::tt_metal::DeviceStorage>) {
                auto device_storage = std::get<tt::tt_metal::DeviceStorage>(tensor.storage());
                if (input_tensor.layout() == Layout::ROW_MAJOR) {
                    if (tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
                        auto device_buffer = device_storage.get_buffer();
                        const auto& tensor_spec = tensor.tensor_spec();
                        auto page_size_bytes = tensor_spec.compute_page_size_bytes();
                        device_buffer->set_page_size(page_size_bytes);
                        return Tensor(
                            std::move(device_storage),
                            new_spec,
                            tensor.distributed_tensor_config(),
                            tensor.tensor_topology());
                    } else {
                        auto device_buffer = device_storage.get_buffer();
                        tt::tt_metal::ShardSpecBuffer shard_spec_buffer = device_buffer->shard_spec();

                        auto shard_spec = shard_spec_buffer.tensor_shard_spec;
                        auto shard_shape = shard_spec.shape;

                        uint32_t mul_div;
                        if (new_logical_shape[-1] == 0 || shard_shape[1] == 0) {
                            mul_div = 0;
                        } else {
                            mul_div = new_logical_shape[-1] > shard_shape[1] ? (new_logical_shape[-1] / shard_shape[1])
                                                                             : (shard_shape[1] / new_logical_shape[-1]);
                        }

                        shard_spec.shape[0] = new_logical_shape[-1] > shard_shape[1] ? shard_shape[0] / mul_div
                                                                                     : shard_shape[0] * mul_div;
                        shard_spec.shape[1] = new_logical_shape[-1];

                        MemoryConfig mem_config = input_tensor.memory_config().with_shard_spec(shard_spec);

                        auto upd_spec = ttnn::TensorSpec(
                            new_logical_shape,
                            TensorLayout::fromPaddedShape(
                                input_tensor.dtype(),
                                input_tensor.tensor_spec().page_config(),
                                mem_config,
                                new_logical_shape,
                                new_padded_shape));

                        shard_spec_buffer.page_shape = {1, new_logical_shape[-1]};
                        shard_spec_buffer.tensor2d_shape_in_pages = {
                            upd_spec.physical_shape().height() / shard_spec_buffer.page_shape[0],
                            upd_spec.physical_shape().width() / shard_spec_buffer.page_shape[1]};
                        shard_spec_buffer.set_shard_spec(shard_spec);
                        device_buffer->set_shard_spec(shard_spec_buffer);

                        auto page_size_bytes = upd_spec.compute_page_size_bytes();
                        device_buffer->set_page_size(page_size_bytes);

                        return Tensor(
                            std::move(device_storage),
                            upd_spec,
                            tensor.distributed_tensor_config(),
                            tensor.tensor_topology());
                    }
                } else {
                    return Tensor(
                        std::move(device_storage),
                        new_spec,
                        tensor.distributed_tensor_config(),
                        tensor.tensor_topology());
                }
            } else if constexpr (std::is_same_v<T, tt::tt_metal::HostStorage>) {
                return Tensor(tensor.storage(), new_spec, tensor.distributed_tensor_config(), tensor.tensor_topology());
            } else {
                static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported storage type");
            }
        },
        input_tensor.storage());
    output = tt::tt_metal::set_tensor_id(output);
    tt::tt_metal::GraphTracker::instance().track_function_end(output);
    return output;
}

ttnn::Tensor ViewOperation::invoke(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    return tensor_reshape(tensor, logical_shape, padded_shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tensor_reshape(tensor, shape, shape);
}

}  // namespace ttnn::operations::experimental::reshape
