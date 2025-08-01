// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/command_queue.hpp>
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& logical_shape);

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation);

ttnn::Tensor bound_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const struct Matmul& parameters,
    const uint8_t& queue_id,
    std::optional<ttnn::Tensor>& optional_output_tensor);

struct MatmulOperation {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        bool transpose_a = false,
        bool transpose_b = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

struct MatmulBatchedWeightsOperation {
    static std::vector<Tensor> invoke(
        const Tensor& input_tensor_a,
        const std::vector<Tensor>& input_tensors_b,
        bool transpose_a = false,
        bool transpose_b = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

struct LinearOperation {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const Tensor>& bias = std::nullopt,
        bool transpose_a = false,
        bool transpose_b = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

struct AddmmOperation {
    static void validate(
        const Tensor& input_tensor, const Tensor& mat1_tensor, const Tensor& mat2_tensor, float alpha, float beta);
    static Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& mat1_tensor,
        const Tensor& mat2_tensor,
        float alpha = 1.0f,
        float beta = 1.0f,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        QueueId queue_id = DefaultQueueId);
};

struct SparseMatmulOperation {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const Tensor& sparsity,
        uint32_t nnz,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

}  // namespace matmul
}  // namespace operations
constexpr auto matmul = ttnn::register_operation<"ttnn::matmul", operations::matmul::MatmulOperation>();
constexpr auto linear = ttnn::register_operation<"ttnn::linear", operations::matmul::LinearOperation>();
constexpr auto matmul_batched_weights =
    ttnn::register_operation<"ttnn::matmul_batched_weights", operations::matmul::MatmulBatchedWeightsOperation>();
constexpr auto addmm = ttnn::register_operation<"ttnn::addmm", operations::matmul::AddmmOperation>();
constexpr auto sparse_matmul =
    ttnn::register_operation<"ttnn::sparse_matmul", operations::matmul::SparseMatmulOperation>();
}  // namespace ttnn
