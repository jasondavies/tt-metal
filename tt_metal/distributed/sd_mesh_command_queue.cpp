// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sd_mesh_command_queue.hpp"
#include "tt_metal/common/thread_pool.hpp"
#include <mesh_device.hpp>
#include <mesh_event.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/graph_tracking.hpp>

namespace tt::tt_metal::distributed {

SDMeshCommandQueue::SDMeshCommandQueue(
    MeshDevice* mesh_device, uint32_t id, std::function<std::lock_guard<std::mutex>()> lock_api_function) :
    MeshCommandQueueBase(mesh_device, id, create_passthrough_thread_pool(), lock_api_function) {}

void SDMeshCommandQueue::write_shard_to_device(
    const MeshBuffer& buffer,
    const MeshCoordinate& device_coord,
    const void* src,
    const std::optional<BufferRegion>& region,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto device_buffer = buffer.get_device_buffer(device_coord);
    auto region_value = region.value_or(BufferRegion(0, device_buffer->size()));
    auto shard_view = device_buffer->view(region_value);

    TT_FATAL(sub_device_ids.empty(), "Sub-device IDs are not supported for slow dispatch");
    if (tt::tt_metal::GraphTracker::instance().hook_write_to_device(&buffer)) {
        return;
    }

    tt::tt_metal::detail::WriteToBuffer(
        *shard_view,
        tt::stl::Span<const uint8_t>(static_cast<const uint8_t*>(src) + region_value.offset, region_value.size));
}

void SDMeshCommandQueue::read_shard_from_device(
    const MeshBuffer& buffer,
    const MeshCoordinate& device_coord,
    void* dst,
    const std::optional<BufferRegion>& region,
    std::unordered_map<IDevice*, uint32_t>&,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto device_buffer = buffer.get_device_buffer(device_coord);
    auto shard_view = device_buffer->view(region.value_or(BufferRegion(0, device_buffer->size())));

    TT_FATAL(sub_device_ids.empty(), "Sub-device IDs are not supported for slow dispatch");
    if (tt::tt_metal::GraphTracker::instance().hook_read_from_device(&buffer)) {
        return;
    }

    tt::tt_metal::detail::ReadFromBuffer(*shard_view, static_cast<uint8_t*>(dst));
}

void SDMeshCommandQueue::submit_memcpy_request(std::unordered_map<IDevice*, uint32_t>&, bool) {}

WorkerConfigBufferMgr& SDMeshCommandQueue::get_config_buffer_mgr(uint32_t index) {
    TT_THROW("Not supported for slow dispatch");
}

void SDMeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    auto lock = lock_api_function_();
    if (!blocking) {
        log_warning(
            tt::LogMetal, "Using Slow Dispatch for {}. This leads to blocking workload exection.", __FUNCTION__);
    }
    for (auto& [coord_range, program] : mesh_workload.get_programs()) {
        for (const auto& coord : coord_range) {
            auto device = mesh_device_->get_device(coord);
            tt_metal::detail::LaunchProgram(device, program);
        }
    }
}

MeshEvent SDMeshCommandQueue::enqueue_record_event(
    tt::stl::Span<const SubDeviceId>, const std::optional<MeshCoordinateRange>& device_range) {
    // No synchronization is needed for slow dispatch, returning a dummy value
    return MeshEvent(0, mesh_device_, id_, device_range.value_or(MeshCoordinateRange(mesh_device_->shape())));
}

MeshEvent SDMeshCommandQueue::enqueue_record_event_to_host(
    tt::stl::Span<const SubDeviceId>, const std::optional<MeshCoordinateRange>& device_range) {
    // No synchronization is needed for slow dispatch, returning a dummy value
    return MeshEvent(0, mesh_device_, id_, device_range.value_or(MeshCoordinateRange(mesh_device_->shape())));
}

void SDMeshCommandQueue::enqueue_wait_for_event(const MeshEvent&) {}
void SDMeshCommandQueue::finish(tt::stl::Span<const SubDeviceId>) {}
void SDMeshCommandQueue::finish_nolock(tt::stl::Span<const SubDeviceId>) {}

void SDMeshCommandQueue::reset_worker_state(bool, uint32_t, const vector_aligned<uint32_t>&) {}

void SDMeshCommandQueue::record_begin(const MeshTraceId&, const std::shared_ptr<MeshTraceDescriptor>&) {
    TT_THROW("Not supported for slow dispatch");
}

void SDMeshCommandQueue::record_end() { TT_THROW("Not supported for slow dispatch"); }

void SDMeshCommandQueue::enqueue_trace(const MeshTraceId&, bool) { TT_THROW("Not supported for slow dispatch"); }

}  // namespace tt::tt_metal::distributed
