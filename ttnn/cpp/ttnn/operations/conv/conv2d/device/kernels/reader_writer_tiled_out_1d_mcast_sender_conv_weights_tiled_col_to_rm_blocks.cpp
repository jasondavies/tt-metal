// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "height_sharded_reader_common.hpp"
#include "debug/debug.h"

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t bias_in_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);

    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(8);
    constexpr uint32_t weight_block_height_ntiles = get_compile_time_arg_val(9);
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(10);
    constexpr uint32_t weight_stride_h = get_compile_time_arg_val(11);
    constexpr uint32_t weight_next_block_stride_h = get_compile_time_arg_val(12);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);

    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);

    // Split reader args
    constexpr bool split_reader = get_compile_time_arg_val(17);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(20);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(21);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(22);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(24);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(25);
    constexpr uint32_t stride_w = get_compile_time_arg_val(26);

    uint32_t i = 0;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i++);
    // Bias arg. Unused if bias fusion is not enabled.
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);

    const uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i++);
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t noop = get_arg_val<uint32_t>(i++);
    if (noop) {
        return;
    }

    if constexpr (split_reader && needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_second_reader>();
    }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;

    // mcast args
    const uint32_t weights_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_dests = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_cores = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr;
    uint32_t reader_idx;
    if constexpr (split_reader) {
        packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));
        reader_idx = 0;
    }

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    *(weights_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* weights_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_sender_semaphore_addr);

    const uint64_t weights_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        weights_mcast_dest_noc_start_x,
        weights_mcast_dest_noc_start_y,
        weights_mcast_dest_noc_end_x,
        weights_mcast_dest_noc_end_y,
        weights_mcast_receiver_semaphore_addr);
#endif

    // read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS

    constexpr uint32_t bias_pagesize = get_tile_size(bias_cb_id);
    constexpr DataFormat bias_df = get_dataformat(bias_cb_id);
    const InterleavedAddrGenFast<bias_in_dram> s_bias = {
        .bank_base_address = bias_addr, .page_size = bias_pagesize, .data_format = bias_df};

    bool load_bias = true;
#endif
    constexpr uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    constexpr DataFormat weight_df = get_dataformat(cb_id_weight);
    const InterleavedAddrGenFast<true> s_weight = {
        .bank_base_address = weight_addr_dram_base, .page_size = weight_tile_nbytes, .data_format = weight_df};

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    constexpr uint32_t weight_inner_block_stride_h =
        weight_next_block_stride_h / weight_block_height_num_outer;  // TODO: Pass as args
    const uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);
    // coalesce reads along weight_size_w
    uint32_t start_reader_idx;
    if constexpr (split_reader) {
        start_reader_idx = (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1;
    }

    for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        // READ WEIGHTS + MCAST SEND WEIGHTS
        // read weight blocks inner dim
        // read weight slice - 1 block of weights in width dim and full weight matrix height
        // read slice only once for all activation blocks
        uint32_t weight_h_offset = 0;

        uint32_t weight_current_block_start_tile_id = 0;

        uint32_t reader_offset = act_l1_read_addr;
        for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
            if constexpr (split_reader) {
                // Do the second half of the reads for act
                noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                reader_idx = start_reader_idx;
                cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles);
                uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
                read_sticks<
                    dilation_w,
                    coalesced_read_bytes,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes,
                    stride_w_bytes,
                    weight_size_w,
                    stride_w>(packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);
                noc_async_read_barrier();
                cb_push_back(cb_id_act_second_reader, act_block_num_tiles);

                reader_offset += window_outer_offset;
            }

            // Do weights read + mcast
            cb_reserve_back(cb_id_weight, weight_block_num_tiles);
            if (bh == 0) {
                uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);
                uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id + weight_h_offset;

                // mcast args
                uint32_t weights_start_address = weight_write_l1_addr;
                uint32_t weights_block_size_bytes = 0;

                // loop over weight block tiles along h
                for (uint32_t weight_tile_h_i = 0; weight_tile_h_i < weight_block_height_ntiles; ++weight_tile_h_i) {
                    uint32_t weight_tile_id = weight_row_start_tile_id;
                    // loop over weight block tiles along w
                    for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                        // DPRINT << "weight_tile_id=" << weight_tile_id << ENDL();
                        noc_async_read_tile(weight_tile_id, s_weight, weight_write_l1_addr);
                        weight_write_l1_addr += weight_tile_nbytes;
                        weights_block_size_bytes += weight_tile_nbytes;
                        weight_tile_id += 1;
                    }  // for weight_block_w
                    weight_row_start_tile_id += weight_stride_h;
                }  // for weight_block_h
                noc_async_read_barrier();

#ifndef SKIP_MCAST
                // wait until all weights mcast destinations have atomically incremented the weights
                // semaphore_addr (i.e. its value should be weights_mcast_num_dests), then reset the
                // semaphore_addr value back to zero for the next block
                noc_semaphore_wait(weights_mcast_sender_semaphore_addr_ptr, weights_mcast_num_dests);
                noc_semaphore_set(weights_mcast_sender_semaphore_addr_ptr, 0);

                // Now we have the block in the CB address, we can mcast to dests!
                uint64_t weights_multicast_data_addr = get_noc_multicast_addr(
                    weights_mcast_dest_noc_start_x,
                    weights_mcast_dest_noc_start_y,
                    weights_mcast_dest_noc_end_x,
                    weights_mcast_dest_noc_end_y,
                    weights_start_address);
                // num_dests must not include source, since we are NOT really doing a local copy!
                noc_async_write_multicast(
                    weights_start_address,
                    weights_multicast_data_addr,
                    weights_block_size_bytes,
                    weights_mcast_num_cores,
                    true);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                // statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and
                // may not be sent in order they are issued
                noc_async_writes_flushed();
#endif

                // We should also multicast the flag to destinations
                // num_dests must not include source, since we are NOT really doing a local copy!
                noc_semaphore_set_multicast(
                    weights_mcast_receiver_semaphore_addr,
                    weights_mcast_receiver_semaphore_noc_addr,
                    weights_mcast_num_cores,
                    false);
#endif

                weight_current_block_start_tile_id += weight_next_block_stride_h;
            }

            cb_push_back(cb_id_weight, weight_block_num_tiles);
        }  // for num_blocks_weight_h
        weight_h_offset += weight_inner_block_stride_h;

#ifdef FUSE_BIAS
        if (load_bias) {
            cb_reserve_back(bias_cb_id, bias_ntiles);
            uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);

            // mcast args
            uint32_t bias_start_address = bias_l1_addr;
            uint32_t bias_block_size_bytes = 0;
            for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles; ++bias_tile) {
                noc_async_read_tile(bias_tile, s_bias, bias_l1_addr);
                bias_l1_addr += bias_pagesize;
                bias_block_size_bytes += bias_pagesize;
            }
            noc_async_read_barrier();

// MCAST BIAS (shares some mcast args with weights)
#ifndef SKIP_MCAST
            // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
            // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to zero
            // for the next block
            noc_semaphore_wait(weights_mcast_sender_semaphore_addr_ptr, weights_mcast_num_dests);
            noc_semaphore_set(weights_mcast_sender_semaphore_addr_ptr, 0);

            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t bias_multicast_data_addr = get_noc_multicast_addr(
                weights_mcast_dest_noc_start_x,
                weights_mcast_dest_noc_start_y,
                weights_mcast_dest_noc_end_x,
                weights_mcast_dest_noc_end_y,
                bias_start_address);
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                bias_start_address, bias_multicast_data_addr, bias_block_size_bytes, weights_mcast_num_cores, true);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc
            // even though cmd bufs are different Also, this only works because we are setting VCs statically (using
            // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
            // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may not
            // be sent in order they are issued
            noc_async_writes_flushed();
#endif

            // We should also multicast the flag to destinations
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_semaphore_set_multicast(
                weights_mcast_receiver_semaphore_addr,
                weights_mcast_receiver_semaphore_noc_addr,
                weights_mcast_num_cores,
                false);
#endif

            cb_push_back(bias_cb_id, bias_ntiles);
            load_bias = false;
        }
#endif
        if constexpr (split_reader) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
        }
    }  // out_num_blocks_h
}
