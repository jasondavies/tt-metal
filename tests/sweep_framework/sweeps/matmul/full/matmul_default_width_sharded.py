# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
from loguru import logger
import functools
import operator

import pytest
import torch

import ttnn

from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.ttnn.utils_for_testing import (
    get_per_core_size_and_num_cores,
    start_measuring_time,
    stop_measuring_time,
)
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return


TIMEOUT = 5


def get_width_sharded_specs(k_size_choices: List[int], num_cores_choices: List[int]) -> Tuple[int, int, int]:
    for k_size in k_size_choices:
        for per_core_width, num_cores_width in get_per_core_size_and_num_cores(
            k_size, num_cores_choices, max_per_core_size=1024
        ):
            yield (k_size, per_core_width, num_cores_width)


parameters = {
    "default": {
        "batch_sizes": [(1,), (2,)],
        "m_size": [x if x > 0 else 32 for x in range(0, 384, 192)],
        "width_sharded_specs": list(
            get_width_sharded_specs(
                [x if x > 0 else 32 for x in range(0, 4096, 384)],
                [x if x > 0 else 1 for x in range(0, 50, 10)],
            )
        ),
        "n_size": [x if x > 0 else 32 for x in range(0, 4096, 384)],
        "batch_matrix_multiply": [False],
        "input_a_memory_config": [ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        "input_a_dtype": [ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16],
        "output_dtype": [ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "compute_kernel_config": [None],
    }
}


def run_matmul(
    device,
    batch_sizes,
    m_size,
    width_sharded_specs,
    n_size,
    batch_matrix_multiply,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    compute_kernel_config,
) -> list:
    k_size, per_core_width, num_cores_width = width_sharded_specs
    total_height = functools.reduce(operator.mul, batch_sizes) * m_size

    core_grid = device.compute_with_storage_grid_size()

    assert input_a_memory_config == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    # TODO: row_wise=False and ROW_MAJOR shard orientation gives bad PCC
    # TODO: COL_MAJOR shard orientation doesn't work for get_matmul_program_config
    input_a_memory_config = input_a_memory_config.with_shard_spec(
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(num_cores_width, core_grid, row_wise=True),
            (total_height, per_core_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
    )

    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

    input_a_layout = input_layout
    input_b_layout = input_layout

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=input_a_layout,
        dtype=input_a_dtype,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=input_b_layout,
        dtype=input_b_dtype,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    op_output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_memory_config,
        dtype=output_dtype,
        compute_kernel_config=compute_kernel_config,
    )
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    expected_pcc = 0.99 if k_size < 1024 else 0.98
    tensors = [input_tensor_a, input_tensor_b, op_output_tensor]
    flop_counts = list(batch_sizes) + [m_size, n_size, 2, k_size]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf, flop_counts)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_matmul(
    device,
    batch_sizes,
    m_size,
    width_sharded_specs,
    n_size,
    batch_matrix_multiply,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    compute_kernel_config,
):
    (result, msg), e2e_perf = run_matmul(
        device,
        batch_sizes,
        m_size,
        width_sharded_specs,
        n_size,
        batch_matrix_multiply,
        input_a_memory_config,
        input_b_memory_config,
        output_memory_config,
        input_a_dtype,
        input_b_dtype,
        output_dtype,
        input_layout,
        compute_kernel_config,
    )
    assert result, msg
    logger.info(f"e2e_perf: {e2e_perf}")


def run(
    batch_sizes,
    m_size,
    width_sharded_specs,
    n_size,
    batch_matrix_multiply,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    compute_kernel_config,
    *,
    device,
) -> list:
    return run_matmul(
        device,
        batch_sizes,
        m_size,
        width_sharded_specs,
        n_size,
        batch_matrix_multiply,
        input_a_memory_config,
        input_b_memory_config,
        output_memory_config,
        input_a_dtype,
        input_b_dtype,
        output_dtype,
        input_layout,
        compute_kernel_config,
    )
