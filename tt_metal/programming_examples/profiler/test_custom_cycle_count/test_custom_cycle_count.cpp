// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

using namespace tt;

bool RunCustomCycle(tt_metal::IDevice* device, int loop_count) {
    bool pass = true;

    CoreCoord compute_with_storage_size = device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores(start_core, end_core);

    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr int loop_size = 50;
    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)}, {"LOOP_SIZE", std::to_string(loop_size)}};

    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = kernel_defines});

    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = kernel_defines});

    std::vector<uint32_t> trisc_kernel_args = {};
    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count_compute.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = trisc_kernel_args, .defines = kernel_defines});

    EnqueueProgram(device->command_queue(), program, false);
    tt_metal::detail::ReadDeviceProfilerResults(device);

    return pass;
}

int main() {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        int loop_count = 2000;
        pass &= RunCustomCycle(device, loop_count);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        fmt::print(stderr, "{}\n", e.what());
        // Capture system call errors that may have returned from driver/kernel
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
