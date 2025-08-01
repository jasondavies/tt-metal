// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <random>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include <tt-metalium/host_buffer.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/device.hpp>
#include <tt_stl/reflection.hpp>
#include "types.hpp"

namespace tt {

namespace tt_metal {

namespace distributed {
class MeshDevice;
class MeshCommandQueue;
}  // namespace distributed

class Tensor {
public:
    std::optional<std::int64_t> tensor_id = std::nullopt;

    // Shared pointer to all attributes associated with this tensor
    // Can be safely passed between threads when the tensor is copied
    std::shared_ptr<TensorAttributes> tensor_attributes = nullptr;

    // Shorthand for checking if this Tensor is allocated on MeshDevice. If set, is never nullptr.
    // If not set, the tensor can either be on host or allocated on a single device.
    // TODO: #21099 - This won't be needed after the migration to MeshDevice is complete.
    std::optional<distributed::MeshDevice*> mesh_device_ = std::nullopt;

    // ======================================================================================
    //                                  Hi Level APIs
    // ======================================================================================
    [[nodiscard]] explicit Tensor() = default;
    [[nodiscard]] Tensor(const Tensor& other);
    [[nodiscard]] Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // Constructs a tensor with `Storage`, `TensorSpec`, and `TensorTopology`.
    [[nodiscard]] Tensor(
        Storage storage,
        TensorSpec tensor_spec,
        DistributedTensorConfig distributed_tensor_config,
        TensorTopology tensor_topology);

    // Constructors of `Tensor` that take physical data encoded in `HostBuffer`.
    // The encoded data type and physical size of the data must match the specified tensor physical shape and data type.
    [[nodiscard]] Tensor(
        HostBuffer buffer,
        const ttnn::Shape& shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);
    [[nodiscard]] Tensor(
        HostBuffer buffer,
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);
    [[nodiscard]] Tensor(HostBuffer buffer, TensorSpec tensor_spec);

    // Converts a buffer of elements of type `T` to a `Tensor`.
    // Elements in the buffer are assumed to be stored in row-major order. The size of the buffer and the type of the
    // elements have to match `spec`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
    //
    // The data in the buffer is copied into a tensor with host storage.
    template <typename T>
    [[nodiscard]] static Tensor from_span(
        tt::stl::Span<const T> buffer,
        const TensorSpec& spec,
        distributed::MeshDevice* device = nullptr,
        ttnn::QueueId cq_id = ttnn::DefaultQueueId,
        T pad_value = 0);

    // Creates a `Tensor` with storage "borrowed" from the buffer of elements of type `T`.
    //
    // The primary use case for this API is to interop with Python, where `MemoryPin` can be set to retain the lifetime
    // of the Python object that owns the underlying data. For example, in pybind11:
    //
    // py::object py_tensor = ...;
    // MemoryPin py_data_pin(std::make_shared<py::object>(py_tensor));
    // Tensor tensor = Tensor::from_borrowed_data(buffer, shape, py_data_pin);
    //
    // This API can also be used to create file-backed Tensors by means of `mmap`:
    //
    // void* mmap_addr = mmap(...);
    // MemoryPin memory_pin(std::shared_ptr<void>(mmap_addr, [](void* addr) { munmap(addr, ...); }));
    // Tensor tensor = Tensor::from_borrowed_data(
    //     tt::stl::Span<T>(reinterpret_cast<T*>(mmap_addr), buffer_size), shape, memory_pin);
    //
    template <typename T>
    [[nodiscard]] static Tensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        tt::tt_metal::MemoryPin buffer_pin,
        const std::optional<Tile>& tile = std::nullopt);

    // Overload that takes `on_creation_callback` and `on_destruction_callback` as separate arguments.
    template <typename T>
    [[nodiscard]] static Tensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        const std::function<void()>& on_creation_callback,
        const std::function<void()>& on_destruction_callback,
        const std::optional<Tile>& tile = std::nullopt) {
        return from_borrowed_data(buffer, shape, MemoryPin(on_creation_callback, on_destruction_callback), tile);
    }

    // Same as `from_span`, but operates on a vector instead.
    template <typename T>
    [[nodiscard]] static Tensor from_vector(
        const std::vector<T>& buffer,
        const TensorSpec& spec,
        distributed::MeshDevice* device = nullptr,
        ttnn::QueueId cq_id = ttnn::DefaultQueueId,
        T pad_value = 0) {
        return from_span(tt::stl::Span<const T>(buffer), spec, device, cq_id, pad_value);
    }

    // Same as `from_vector`, but takes in an rvalue. No copies will be made, if the target layout is row-major,
    // physical shape matches logical shape, and no type conversion is needed.
    template <typename T>
    [[nodiscard]] static Tensor from_vector(
        std::vector<T>&& buffer,
        const TensorSpec& spec,
        distributed::MeshDevice* device = nullptr,
        ttnn::QueueId cq_id = ttnn::DefaultQueueId,
        T pad_value = 0);

    // Converts a `Tensor` to a `std::vector<T>`.
    // Elements in the vector will be stored in row-major order. The type of the requested vector has to match that of
    // the `Tensor`; block float formats such as BFLOAT8_B and BFLOAT4_B require `T` equal `float`.
    //
    // If the tensor resides on a device, it will be brough back to host.
    template <typename T>
    [[nodiscard]] std::vector<T> to_vector(ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    template <typename T>
    [[nodiscard]] T item(ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    [[nodiscard]] Tensor to_device(
        distributed::MeshDevice* mesh_device,
        const MemoryConfig& mem_config = MemoryConfig{},
        ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    [[nodiscard]] Tensor to_layout(Layout target_layout) const;

    [[nodiscard]] Tensor pad(
        const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const;

    [[nodiscard]] Tensor cpu(bool blocking = true, ttnn::QueueId cq_id = ttnn::DefaultQueueId) const;

    [[nodiscard]] Tensor unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const;

    [[nodiscard]] Tensor pad_to_tile(float pad_value) const;

    [[nodiscard]] Tensor unpad_from_tile(const ttnn::Shape& output_tensor_shape) const;

    [[nodiscard]] std::string write_to_string() const;
    void print() const;

    // Deallocates device-side Tensor storage.
    // If the tensor is on host, does nothing.
    void deallocate(bool force = false);

    [[nodiscard]] Tensor extract_shard(const CoreCoord& core) const;
    [[nodiscard]] Tensor extract_shard(const uint32_t& core_id) const;

    // ======================================================================================
    //                                  Low Level APIs
    // ======================================================================================
    [[nodiscard]] Tensor reshape(const ttnn::Shape& new_shape) const;
    [[nodiscard]] Tensor reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const;
    // ======================================================================================
    //                                      Getters
    // ======================================================================================
    const Storage& storage() const;
    Storage& storage();
    DataType dtype() const;
    Layout layout() const;
    const ttnn::Shape& logical_shape() const;
    const ttnn::Shape& padded_shape() const;
    const TensorSpec& tensor_spec() const;
    uint64_t logical_volume() const;
    uint64_t physical_volume() const;
    const DistributedTensorConfig& distributed_tensor_config() const;
    const MemoryConfig& memory_config() const;

    // Multi-device topology configuration - tracks how tensor is distributed across mesh devices
    const TensorTopology& tensor_topology() const;

    // For sharded tensors, at least one of ShardSpec or NdShardSpec will be provided.
    const std::optional<ShardSpec>& shard_spec() const;
    const std::optional<NdShardSpec>& nd_shard_spec() const;

    // ======================================================================================
    //                                      Extra Helper Functions
    // ======================================================================================
    StorageType storage_type() const;
    ttnn::Shape strides() const;

    bool is_scalar() const;

    bool is_allocated() const;

    // Returns device `Buffer`.
    // Throws if the tensor is not allocated on a device.
    Buffer* buffer() const;

    // Returns device `Storage`.
    // Throws if the tensor is not on device.
    const DeviceStorage& device_storage() const&;
    const DeviceStorage& device_storage() const&& = delete;  // prevents dangling reference to temporaries.

    // Returns host `Storage`.
    // Throws if the tensor is not on host.
    const HostStorage& host_storage() const&;
    const HostStorage& host_storage() const&& = delete;  // prevents dangling reference to temporaries.

    // Returns device `MeshBuffer`.
    // Throws if the tensor is not allocated on a device.
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer() const;

    // Returns the device the tensor is allocated on.
    // Throws if the tensor is not allocated on a device.
    distributed::MeshDevice* device() const;

    // NOTE: Keeping this for backward compatibility.
    // This is deprecated
    distributed::MeshDevice* mesh_device() const { return device(); }

    bool is_sharded() const;

    // Size in bytes of a single element held in tensor
    uint32_t element_size() const;

    static constexpr auto attribute_names = std::forward_as_tuple("storage", "tensor_spec");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->tensor_attributes->get_storage(), this->tensor_attributes->get_tensor_spec());
    }

private:
    void init(
        Storage storage,
        TensorSpec tensor_spec,
        DistributedTensorConfig distributed_tensor_config,
        TensorTopology tensor_topology);
    void deallocate_impl(bool force);
};

Tensor create_device_tensor(const TensorSpec& tensor_spec, IDevice* device);

[[deprecated]]
Tensor create_device_tensor(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    IDevice* device,
    const MemoryConfig& memory_config = MemoryConfig{},
    const std::optional<Tile>& tile = std::nullopt);

// The set of memcpy functions below are used to copy data between host buffers/tensors and single-device tensors
void memcpy(
    CommandQueue& queue,
    void* dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);
void memcpy(
    distributed::MeshCommandQueue& queue,
    void* dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void memcpy(
    CommandQueue& queue, Tensor& dst, const void* src, const std::optional<BufferRegion>& region = std::nullopt);
void memcpy(
    distributed::MeshCommandQueue& queue,
    Tensor& dst,
    const void* src,
    const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    CommandQueue& queue, Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt);
void memcpy(
    distributed::MeshCommandQueue& queue,
    Tensor& dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    void* dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt, bool blocking = true);

void memcpy(Tensor& dst, const void* src, const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt);

// Allocates a tensor on device.
Tensor allocate_tensor_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);

// Allocates a tensor on host. Uses `mesh_device` to allocate sufficient number of host buffers for each multi-device
// shard.
Tensor allocate_tensor_on_host(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);

// Writes tensor from `src` to `dst`; supports only host-to-device and device-to-host transfers.
void write_tensor(const Tensor& src, Tensor& dst, bool blocking = true, ttnn::QueueId cq_id = ttnn::DefaultQueueId);

Tensor set_tensor_id(const Tensor& tensor);

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;
using TensorSpec = tt::tt_metal::TensorSpec;

}  // namespace ttnn
