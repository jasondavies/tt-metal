// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tt_metal/impl/sub_device/sub_device_manager.hpp"

#include "tt_metal/common/assert.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/sub_device/sub_device.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/tt_stl/span.hpp"

namespace tt::tt_metal {

namespace detail {

SubDeviceManager::SubDeviceManager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size, Device *device) :
    sub_devices_(sub_devices.begin(), sub_devices.end()),
    local_l1_size_(align(local_l1_size, hal.get_alignment(HalMemType::L1))),
    device_(device) {
    TT_ASSERT(device != nullptr, "Device must not be null");
    this->validate_sub_devices();
    this->populate_sub_device_ids();
    this->populate_num_cores();
    this->populate_sub_allocators();
    this->populate_noc_data();
    this->populate_worker_launch_message_buffer_state();
}

SubDeviceManager::SubDeviceManager(Device *device, std::unique_ptr<Allocator> &&global_allocator) : device_(device) {
    TT_ASSERT(device != nullptr, "Device must not be null");
    this->local_l1_size_ = 0;
    const auto& compute_grid_size = this->device_->compute_with_storage_grid_size();
    const auto& active_eth_cores = this->device_->get_active_ethernet_cores(true);
    std::vector<CoreRange> active_eth_core_ranges;
    active_eth_core_ranges.reserve(active_eth_cores.size());
    for (const auto& core : active_eth_cores) {
        active_eth_core_ranges.emplace_back(core, core);
    }

    this->sub_devices_ = {SubDevice(std::array{
        CoreRangeSet(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1})),
        CoreRangeSet(std::move(active_eth_core_ranges))})};
    this->populate_sub_device_ids();
    // No need to validate sub-devices since this constructs a sub-device of the entire grid
    this->populate_num_cores();
    this->sub_device_allocators_.push_back(std::move(global_allocator));
    this->populate_noc_data();
    this->populate_worker_launch_message_buffer_state();
}

SubDeviceManager::~SubDeviceManager() {
    for (const auto &allocator : this->sub_device_allocators_) {
        if (allocator) {
            // Clear the bank managers, this makes subsequent buffer deallocations fast
            allocator::clear(*allocator);
            // Deallocate all buffers
            // This is done to set buffer object status to Deallocated
            const auto &allocated_buffers = allocator::get_allocated_buffers(*allocator);
            for (auto buf = allocated_buffers.begin(); buf != allocated_buffers.end();) {
                tt::tt_metal::DeallocateBuffer(*(*(buf++)));
            }
        }
    }
}

uint8_t SubDeviceManager::num_sub_devices() const { return this->sub_devices_.size(); }

const std::vector<SubDeviceId> &SubDeviceManager::get_sub_device_ids() const {
    return this->sub_device_ids_;
}

const SubDevice& SubDeviceManager::sub_device(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return sub_devices_[sub_device_index];
}

const vector_memcpy_aligned<uint32_t> &SubDeviceManager::noc_mcast_unicast_data() const {
    return noc_mcast_unicast_data_;
}

uint8_t SubDeviceManager::num_noc_mcast_txns(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return this->num_noc_mcast_txns_[sub_device_index];
}

uint8_t SubDeviceManager::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return this->num_noc_unicast_txns_[sub_device_index];
}

uint8_t SubDeviceManager::noc_mcast_data_start_index(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return this->noc_mcast_data_start_index_[sub_device_index];
}

uint8_t SubDeviceManager::noc_unicast_data_start_index(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return this->noc_unicast_data_start_index_[sub_device_index];
}

const std::unique_ptr<Allocator> &SubDeviceManager::get_initialized_allocator(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    TT_FATAL(this->sub_device_allocators_[sub_device_index], "SubDevice allocator not initialized");
    return this->sub_device_allocators_[sub_device_index];
}

std::unique_ptr<Allocator> &SubDeviceManager::sub_device_allocator(SubDeviceId sub_device_id) {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return this->sub_device_allocators_[sub_device_index];
}

std::shared_ptr<TraceBuffer> &SubDeviceManager::create_trace(uint32_t tid) {
    auto [trace, emplaced] = this->trace_buffer_pool_.emplace(tid, Trace::create_empty_trace_buffer());
    TT_ASSERT(emplaced, "Trace buffer with tid {} already exists", tid);
    return trace->second;
}

void SubDeviceManager::release_trace(uint32_t tid) {
    this->trace_buffer_pool_.erase(tid);
}

std::shared_ptr<TraceBuffer> SubDeviceManager::get_trace(uint32_t tid) {
    auto trace = this->trace_buffer_pool_.find(tid);
    if (trace != this->trace_buffer_pool_.end()) {
        return trace->second;
    }
    return nullptr;
}

void SubDeviceManager::reset_worker_launch_message_buffer_state() {
    std::for_each(this->worker_launch_message_buffer_state_.begin(), this->worker_launch_message_buffer_state_.end(), std::mem_fn(&LaunchMessageRingBufferState::reset));
}

LaunchMessageRingBufferState& SubDeviceManager::get_worker_launch_message_buffer_state(SubDeviceId sub_device_id) {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return this->worker_launch_message_buffer_state_[sub_device_index];
}

bool SubDeviceManager::has_allocations() const {
    for (const auto& allocator : this->sub_device_allocators_) {
        if (allocator && allocator->allocated_buffers.size() > 0) {
            return true;
        }
    }
    return false;
}

DeviceAddr SubDeviceManager::local_l1_size() const { return this->local_l1_size_; }

uint8_t SubDeviceManager::get_sub_device_index(SubDeviceId sub_device_id) const {
    auto sub_device_index = sub_device_id.to_index();
    TT_FATAL(
        sub_device_index < this->sub_devices_.size(),
        "SubDevice index {} out of bounds {}",
        sub_device_index,
        this->sub_devices_.size());
    return sub_device_index;
}

void SubDeviceManager::validate_sub_devices() const {
    TT_FATAL(this->sub_devices_.size() <= SubDeviceManager::MAX_NUM_SUB_DEVICES, "Too many sub devices specified");
    // Validate sub device cores fit inside the device grid
    const auto& compute_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange device_worker_cores = CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1});
    const auto& device_eth_cores = this->device_->get_active_ethernet_cores(true);
    for (const auto& sub_device : this->sub_devices_) {
        const auto& worker_cores = sub_device.cores(HalProgrammableCoreType::TENSIX);
        TT_FATAL(
            device_worker_cores.contains(worker_cores),
            "Tensix cores {} specified in sub device must be within device grid {}",
            worker_cores,
            device_worker_cores);
        const auto& eth_cores = sub_device.cores(HalProgrammableCoreType::ACTIVE_ETH);
        uint32_t num_eth_cores = 0;
        for (const auto& dev_eth_core : device_eth_cores) {
            if (eth_cores.contains(dev_eth_core)) {
                num_eth_cores++;
            }
        }
        TT_FATAL(
            num_eth_cores == eth_cores.num_cores(),
            "Ethernet cores {} specified in sub device must be within device grid",
            eth_cores);
    }
    if (this->sub_devices_.size() < 2) {
        return;
    }
    // Validate no overlap of sub devices
    for (uint32_t i = 0; i < this->sub_devices_.size(); ++i) {
        for (uint32_t j = i + 1; j < this->sub_devices_.size(); ++j) {
            for (uint32_t k = 0; k < NumHalProgrammableCoreTypes; ++k) {
                TT_FATAL(
                    !(this->sub_devices_[i].cores()[k].intersects(this->sub_devices_[j].cores()[k])),
                    "SubDevices specified for SubDeviceManager intersect");
            }
        }
    }
}

void SubDeviceManager::populate_sub_device_ids() {
    this->sub_device_ids_.resize(this->num_sub_devices());
    for (uint8_t i = 0; i < this->num_sub_devices(); ++i) {
        this->sub_device_ids_[i] = SubDeviceId{i};
    }
}

void SubDeviceManager::populate_num_cores() {
    for (const auto& sub_device : this->sub_devices_) {
        for (uint32_t i = 0; i < NumHalProgrammableCoreTypes; ++i) {
            this->num_cores_[i] += sub_device.num_cores(static_cast<HalProgrammableCoreType>(i));
        }
    }
}

void SubDeviceManager::populate_sub_allocators() {
    this->sub_device_allocators_.resize(this->num_sub_devices());
    if (this->local_l1_size_ == 0) {
        return;
    }
    const auto& global_allocator_config = this->device_->get_initialized_allocator()->config;
    // Construct allocator config from soc_desc
    // Take max alignment to satisfy NoC rd/wr constraints
    // Tensix/Eth -> PCIe/DRAM src and dst addrs must be L1_ALIGNMENT aligned
    // PCIe/DRAM -> Tensix/Eth src and dst addrs must be DRAM_ALIGNMENT aligned
    // Tensix/Eth <-> Tensix/Eth src and dst addrs must be L1_ALIGNMENT aligned
    for (uint32_t i = 0; i < this->num_sub_devices(); ++i) {
        const auto& compute_cores = this->sub_devices_[i].cores(HalProgrammableCoreType::TENSIX);
        if (compute_cores.empty()) {
            continue;
        }
        AllocatorConfig config(
            {.num_dram_channels = global_allocator_config.num_dram_channels,
             .dram_bank_size = 0,
             .dram_bank_offsets = global_allocator_config.dram_bank_offsets,
             .dram_unreserved_base = global_allocator_config.dram_unreserved_base,
             .l1_unreserved_base = global_allocator_config.l1_unreserved_base,
             .worker_grid = compute_cores,
             .worker_l1_size = global_allocator_config.l1_unreserved_base + this->local_l1_size_,
             .storage_core_bank_size = std::nullopt,
             .l1_small_size = 0,
             .trace_region_size = 0,
             .core_type_from_noc_coord_table = {},  // Populated later
             .worker_log_to_physical_routing_x = global_allocator_config.worker_log_to_physical_routing_x,
             .worker_log_to_physical_routing_y = global_allocator_config.worker_log_to_physical_routing_y,
             .l1_bank_remap = {},
             .compute_grid = compute_cores,
             .alignment = global_allocator_config.alignment,
             .disable_interleaved = true});
        TT_FATAL(
            config.l1_small_size < (config.storage_core_bank_size.has_value()
                                        ? config.storage_core_bank_size.value()
                                        : config.worker_l1_size - config.l1_unreserved_base),
            "Reserved size must be less than bank size");
        TT_FATAL(
            config.l1_small_size % config.alignment == 0,
            "Reserved size must be aligned to allocator alignment {}",
            config.alignment);

        // sub_devices only have compute cores for allocation
        for (const CoreCoord& core : corerange_to_cores(compute_cores)) {
            const auto noc_coord = this->device_->worker_core_from_logical_core(core);
            config.core_type_from_noc_coord_table.insert({noc_coord, AllocCoreType::ComputeAndStore});
        }

        // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
        // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
        TT_ASSERT(this->device_->allocator_scheme_ == MemoryAllocator::L1_BANKING);
        this->sub_device_allocators_[i] = std::make_unique<L1BankingAllocator>(config);
    }
}

void SubDeviceManager::populate_noc_data() {
    uint32_t num_sub_devices = this->num_sub_devices();
    this->num_noc_mcast_txns_.resize(num_sub_devices);
    this->num_noc_unicast_txns_.resize(num_sub_devices);
    this->noc_mcast_data_start_index_.resize(num_sub_devices);
    this->noc_unicast_data_start_index_.resize(num_sub_devices);

    NOC noc_index = this->device_->dispatch_go_signal_noc();
    uint32_t idx = 0;
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        const auto& tensix_cores = this->sub_devices_[i].cores(HalProgrammableCoreType::TENSIX);
        const auto& eth_cores = this->sub_devices_[i].cores(HalProgrammableCoreType::ACTIVE_ETH);

        this->noc_mcast_data_start_index_[i] = idx;
        this->num_noc_mcast_txns_[i] = tensix_cores.size();
        this->noc_mcast_unicast_data_.resize(idx + this->num_noc_mcast_txns_[i] * 2);
        for (const auto& core_range : tensix_cores.ranges()) {
            auto physical_start =
                this->device_->physical_core_from_logical_core(core_range.start_coord, CoreType::WORKER);
            auto physical_end = this->device_->physical_core_from_logical_core(core_range.end_coord, CoreType::WORKER);
            auto physical_core_range = CoreRange(physical_start, physical_end);
            this->noc_mcast_unicast_data_[idx++] = this->device_->get_noc_multicast_encoding(noc_index, physical_core_range);
            this->noc_mcast_unicast_data_[idx++] = core_range.size();
        }
        this->noc_unicast_data_start_index_[i] = idx;

        // TODO: Precompute number of eth cores and resize once
        for (const auto& core_range : eth_cores.ranges()) {
            this->noc_mcast_unicast_data_.resize(idx + core_range.size());
            for (const auto& core : core_range) {
                auto physical_core = this->device_->physical_core_from_logical_core(core, CoreType::ETH);
                this->noc_mcast_unicast_data_[idx++] = this->device_->get_noc_unicast_encoding(noc_index, physical_core);
            }
        }
        this->num_noc_unicast_txns_[i] = idx - this->noc_unicast_data_start_index_[i];

        TT_FATAL(idx <= dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES, "NOC data entries {} exceeds maximum supported size {}", idx, dispatch_constants::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES);
    }
}

void SubDeviceManager::populate_worker_launch_message_buffer_state() {
    this->worker_launch_message_buffer_state_.resize(this->num_sub_devices());
    this->reset_worker_launch_message_buffer_state();
}

}  // namespace detail

}  // namespace tt::tt_metal