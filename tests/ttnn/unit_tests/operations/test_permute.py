# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import itertools

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from models.utility_functions import is_blackhole, skip_for_wormhole_b0

import torch
from collections import defaultdict
from typing import List, Tuple, Dict, Set


def find_tensor_swaps(original: torch.Tensor, modified: torch.Tensor, tolerance: float = 1e-6) -> Dict:
    """
    Find which leaves were swapped between two tensors.

    Args:
        original: Original tensor of shape (96, 96, 96, 96)
        modified: Modified tensor with potential swaps
        tolerance: Tolerance for floating point comparison

    Returns:
        Dictionary with swap analysis results
    """
    assert original.shape == modified.shape == (96, 96, 96, 96), "Tensors must be (96, 96, 96, 96)"

    # Step 1: Find all positions where leaves don't match
    print("Step 1: Finding mismatched positions...")
    leaf_matches = torch.allclose(original, modified, atol=tolerance, rtol=tolerance)
    if leaf_matches:
        return {"swaps": [], "corrupted": [], "total_mismatches": 0}

    # Compare each leaf individually
    mismatched_positions = []
    for i in range(96):
        for j in range(96):
            for k in range(96):
                if not torch.allclose(original[i, j, k, :], modified[i, j, k, :], atol=tolerance, rtol=tolerance):
                    mismatched_positions.append((i, j, k))

    print(f"Found {len(mismatched_positions)} mismatched leaves")

    if len(mismatched_positions) == 0:
        return {"swaps": [], "corrupted": [], "total_mismatches": 0}

    # Step 2: Build lookup table for original leaves
    print("Step 2: Building lookup table...")
    original_leaf_map = {}
    for i in range(96):
        for j in range(96):
            for k in range(96):
                leaf = original[i, j, k, :]
                # Convert to tuple for hashing (approximate for floating point)
                leaf_key = tuple(leaf.round(decimals=8).tolist())
                if leaf_key not in original_leaf_map:
                    original_leaf_map[leaf_key] = []
                original_leaf_map[leaf_key].append((i, j, k))

    # Step 3: For each mismatched position, find where its leaf came from
    print("Step 3: Analyzing swaps...")
    swap_pairs = []
    corrupted_positions = []
    processed_positions = set()

    for pos in mismatched_positions:
        if pos in processed_positions:
            continue

        i, j, k = pos
        modified_leaf = modified[i, j, k, :]
        modified_leaf_key = tuple(modified_leaf.round(decimals=8).tolist())

        # Check if this modified leaf exists in the original tensor
        if modified_leaf_key in original_leaf_map:
            # Find potential source positions
            potential_sources = original_leaf_map[modified_leaf_key]

            # Check which source position is also mismatched (indicating a swap)
            for source_pos in potential_sources:
                if source_pos in mismatched_positions and source_pos not in processed_positions:
                    # Check if source position now contains our original leaf
                    source_i, source_j, source_k = source_pos
                    original_leaf_at_pos = original[i, j, k, :]
                    modified_leaf_at_source = modified[source_i, source_j, source_k, :]

                    if torch.allclose(original_leaf_at_pos, modified_leaf_at_source, atol=tolerance, rtol=tolerance):
                        # Confirmed swap!
                        swap_pairs.append((pos, source_pos))
                        processed_positions.add(pos)
                        processed_positions.add(source_pos)
                        break

            # If no swap partner found, might be a move (not a swap)
            if pos not in processed_positions:
                # This leaf moved from somewhere else, but its original position
                # might have been filled with something else (corruption or complex swap)
                corrupted_positions.append(pos)
                processed_positions.add(pos)
        else:
            # This leaf doesn't exist in original - it's corrupted
            corrupted_positions.append(pos)
            processed_positions.add(pos)

    # Step 4: Verify swaps
    print("Step 4: Verifying swaps...")
    verified_swaps = []
    for pos1, pos2 in swap_pairs:
        i1, j1, k1 = pos1
        i2, j2, k2 = pos2

        # Double check the swap
        orig1 = original[i1, j1, k1, :]
        orig2 = original[i2, j2, k2, :]
        mod1 = modified[i1, j1, k1, :]
        mod2 = modified[i2, j2, k2, :]

        if torch.allclose(orig1, mod2, atol=tolerance, rtol=tolerance) and torch.allclose(
            orig2, mod1, atol=tolerance, rtol=tolerance
        ):
            verified_swaps.append((pos1, pos2))

    # Summary
    results = {
        "total_mismatches": len(mismatched_positions),
        "swaps": verified_swaps,
        "corrupted": corrupted_positions,
        "swap_count": len(verified_swaps),
        "corruption_count": len(corrupted_positions),
    }

    return results


def print_swap_analysis(results: Dict):
    """Print a detailed analysis of the swap detection results"""
    print("\n" + "=" * 60)
    print("TENSOR SWAP ANALYSIS RESULTS")
    print("=" * 60)

    print(f"Total mismatched leaves: {results['total_mismatches']}")
    print(f"Confirmed swaps: {results['swap_count']} pairs ({results['swap_count']*2} positions)")
    print(f"Corrupted/moved leaves: {results['corruption_count']}")

    if results["swaps"]:
        print(f"\nDetected swaps:")
        for i, (pos1, pos2) in enumerate(results["swaps"], 1):
            print(f"  {i}. {pos1} ↔ {pos2}")

    if results["corrupted"]:
        print(f"\nCorrupted/moved positions (first 10):")
        for pos in results["corrupted"][:10]:
            print(f"  - {pos}")
        if len(results["corrupted"]) > 10:
            print(f"  ... and {len(results['corrupted']) - 10} more")


# Example usage and testing
def create_test_tensors():
    """Create test tensors with known swaps for validation"""
    print("Creating test tensors...")

    # Create original random tensor
    torch.manual_seed(42)  # For reproducibility
    original = torch.randn(96, 96, 96, 96)

    # Create modified tensor with some swaps
    modified = original.clone()

    # Perform some known swaps
    known_swaps = [((0, 0, 0), (1, 1, 1)), ((5, 10, 15), (20, 25, 30)), ((50, 60, 70), (80, 85, 90))]

    for (i1, j1, k1), (i2, j2, k2) in known_swaps:
        # Swap the leaves
        temp = modified[i1, j1, k1, :].clone()
        modified[i1, j1, k1, :] = modified[i2, j2, k2, :]
        modified[i2, j2, k2, :] = temp

    # Add some corruption
    modified[10, 20, 30, :] = torch.randn(96) * 1000  # Completely different values

    print(f"Created tensors with {len(known_swaps)} known swaps and 1 corruption")
    return original, modified, known_swaps


# Test the function
if __name__ == "__main__":
    # Create test case
    original, modified, known_swaps = create_test_tensors()

    # Run analysis
    results = find_tensor_swaps(original, modified)

    # Print results
    print_swap_analysis(results)

    # Verify against known swaps
    print(f"\nValidation:")
    print(f"Known swaps: {known_swaps}")
    print(f"Detected swaps: {results['swaps']}")

    detected_set = {tuple(sorted([pos1, pos2])) for pos1, pos2 in results["swaps"]}
    known_set = {tuple(sorted([pos1, pos2])) for pos1, pos2 in known_swaps}

    if detected_set == known_set:
        print("✅ Perfect detection!")
    else:
        print("❌ Detection mismatch")
        print(f"Missing: {known_set - detected_set}")
        print(f"False positives: {detected_set - known_set}")


def find_swapped_leaves(original, modified):
    """Find positions where 96-element leaves differ between tensors"""
    # Compare each leaf (last dimension)
    leaf_matches = torch.all(original == modified, dim=-1)

    # Find positions where leaves don't match
    swapped_positions = torch.where(~leaf_matches)

    return swapped_positions


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        # tensor = torch.full(shape, 1, dtype=torch.bfloat16)
        # tensor[..., 1::2] = 0
        # leaf_values = torch.arange(96, dtype=torch.bfloat16)
        # tensor = leaf_values.expand(shape).clone()
        # return tensor
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 1, 3, 2))

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 1.0)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_transpose(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_on_4D_tensor_with_smaller_tuple_size(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    with pytest.raises(
        RuntimeError,
        match="The number of dimensions in the tensor input does not match the length of the desired ordering",
    ) as exception:
        ttnn.permute(input_tensor, (0, 1, 2))


@pytest.mark.parametrize(
    "perm", [(0,), (0, 1), (1, 0), (0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0)]
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_on_less_than_4D(device, perm, dtype):
    torch.manual_seed(2005)
    shape = tuple([32 * (value + 1) for value in perm])
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, perm)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("s", [8])
@pytest.mark.parametrize("h", [1500])
@pytest.mark.parametrize("w", [64])
# @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
# #24347: Looks like we have a non-det issue on N150 + N300
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_permute_for_specific_case(device, b, s, h, w, dtype):
    torch.manual_seed(2005)
    shape = (b, s, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 1, 3, 2))
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


def test_add_after_permute(device):
    torch.manual_seed(2005)
    torch_a = torch.randn(2, 1280, 8, 8)
    torch_b = torch.randn(1, 1, 2, 1280)
    torch_b_permuted = torch.permute(torch_b, (2, 3, 0, 1))
    torch_output = torch_a + torch_b_permuted

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.permute(b, (2, 3, 0, 1))
    output = a + b
    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output, output, 1.0)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_negative_dim(device, h, w, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, h, w)
    torch_input_tensor = random_torch_tensor(dtype, shape)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, -3, -1, -2))

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, -3, -1, -2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 1.0)


def test_permute_bfloat8(device):
    torch.manual_seed(2005)
    input_a = torch.randn(1, 160, 32, 32)
    torch_output = torch.permute(input_a, (0, 2, 3, 1))

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_output = ttnn.permute(tt_input, (0, 2, 3, 1))
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 1.0)


@pytest.mark.parametrize(
    "shape", [(8, 2, 2, 3, 4), [1, 1370, 1, 3, 1280], [1, 197, 1, 3, 1024], [1, 197, 1, 3, 768], [1, 50, 1, 3, 1024]]
)
@pytest.mark.parametrize("perm", [(0, 3, 2, 1, 4), (3, 1, 2, 0, 4), (0, 3, 2, 1, 4), (1, 3, 2, 0, 4), (0, 3, 1, 2, 4)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_5d(device, shape, perm, dtype):
    torch.manual_seed(2005)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype)

    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 1.0)


@pytest.mark.parametrize("pad_value", [float("-inf"), None])
def test_permute_pad_value(device, pad_value):
    if pad_value is not None and is_blackhole():
        pytest.skip("Blackhole reduce is needed for the full test to work")
    torch.manual_seed(2005)
    input_a = torch.randn((2, 11, 33, 17), dtype=torch.bfloat16)
    torch_output = torch.permute(input_a, (3, 2, 1, 0))

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output = ttnn.permute(tt_input, (3, 2, 1, 0), pad_value=pad_value)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 1.0)


def generate_permutations(N):
    """
    Generator function that yields all permutations of tuples with values 0 to N-1.

    :param N: The number defining the range of values (0 to N-1).
    :yield: Tuples representing each permutation.
    """
    for perm in itertools.permutations(range(N)):
        yield perm


@pytest.mark.parametrize("shape", [(7, 7, 7, 7, 7)])
@pytest.mark.parametrize("perm", generate_permutations(5))
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_permute_5d_width(device, shape, perm, memory_config, dtype):
    torch.manual_seed(2005)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(
        input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config
    )

    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 1.0)


@pytest.mark.parametrize("shape", [(3, 65, 3, 3, 65), (1, 6, 256, 20, 50), (6, 20, 50, 1, 256)])
@pytest.mark.parametrize("perm", [(4, 0, 3, 2, 1), (1, 3, 4, 0, 2), (3, 0, 4, 1, 2)])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    torch.manual_seed(520)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(
        input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config
    )

    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 1.0)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_nd(device, dtype):
    torch.manual_seed(2005)
    shape = (1, 3, 16, 16, 16, 16)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 2, 4, 3, 5, 1))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 2, 4, 3, 5, 1))
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_squeeze(device, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, 3)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(output_tensor, ttnn.to_torch(input_tensor), 1.0)


@pytest.mark.parametrize("shape", [(1, 49, 768)])
@pytest.mark.parametrize("perm", generate_permutations(3))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_3D(device, shape, perm, layout, memory_config, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=dtype, memory_config=memory_config)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_nil_volume_permute(device, dtype):
    torch.manual_seed(2005)
    shape = (1, 0, 30, 32)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 1, 3, 2))
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_5d_tiled_basic(device, dtype):
    torch.manual_seed(2005)
    shape = (10, 10, 10, 100, 100)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (2, 1, 0, 3, 4))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (2, 1, 0, 3, 4))
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_5d_tiled_swap(device, dtype):
    torch.manual_seed(2005)
    shape = (10, 10, 10, 100, 100)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (2, 1, 0, 4, 3))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (2, 1, 0, 4, 3))
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_4d_cn(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (1, 0, 2, 3))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (1, 0, 2, 3))
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_4d_wh(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 1, 3, 2))
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_4d_cnwh(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (1, 0, 3, 2))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (1, 0, 3, 2))
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("shape", [[2, 2, 2, 2, 2, 2, 32, 32]])
@pytest.mark.parametrize("dims", [(5, 4, 3, 2, 1, 0, 7, 6), (5, 4, 3, 2, 1, 0, 6, 7)])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_8d_swapped(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, dims)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, dims)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_identity(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 2, 3))
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, (0, 1, 2, 3))
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("shape", [[2, 2, 67, 67, 65]])
@pytest.mark.parametrize("perm", [(0, 1, 3, 2, 4)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_permute_5d_xh_pad(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


def generate_fixed_w_permutations(N):
    perms_Nd = generate_permutations(N - 1)
    for perm in perms_Nd:
        yield perm + (N - 1,)


@pytest.mark.parametrize("shape", [[7, 7, 7, 33, 33]])
@pytest.mark.parametrize("perm", generate_fixed_w_permutations(5))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_permutations_5d_fixed_w(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("shape", [[1, 9, 91, 7, 9]])
@pytest.mark.parametrize("perm", [[0, 3, 4, 1, 2]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_adversarial(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 32, 32], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("perm", generate_fixed_w_permutations(4))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_permute_4d_fixed_w(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


def generate_fixed_no_dim0_dim1_transpose_permutations(N, dim0, dim1):
    perms_Nd = generate_permutations(N)
    for perm in perms_Nd:
        if perm[dim0] != dim1:
            yield perm


@pytest.mark.parametrize("shape", [[7, 7, 7, 17, 17]])
@pytest.mark.parametrize("perm", [[0, 1, 4, 3, 2]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("pad_value", [35.0, float("-inf"), None])
def test_permute_5d_yw_padded(device, shape, perm, dtype, pad_value):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    ttnn_output = ttnn.permute(input_tensor, perm, pad_value=pad_value)
    output_tensor = ttnn.to_torch(ttnn_output)
    torch_output = torch.permute(torch_tensor, perm)

    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)

    if pad_value != None:
        logical_shape = torch_output.shape
        output_padded = ttnn.from_device(ttnn_output).to_torch()
        padded_shape = output_padded.shape
        num_padded_values = torch.prod(torch.tensor(padded_shape)) - torch.prod(torch.tensor(logical_shape))
        assert torch.sum(output_padded == pad_value) == num_padded_values


@pytest.mark.parametrize("shape", [[33, 1, 17, 33, 33]])
@pytest.mark.parametrize("perm", generate_fixed_no_dim0_dim1_transpose_permutations(5, 4, 3))
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_5d_yw_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 1, 128, 128], [32, 32, 32, 32], [96, 96, 96, 96]])
@pytest.mark.parametrize("perm", [[0, 3, 2, 1], [3, 1, 2, 0], [1, 3, 2, 0], [3, 0, 2, 1]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_4d_yw_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 1, 128, 128], [32, 32, 32, 32], [96, 96, 96, 96]])
@pytest.mark.parametrize("perm", [[2, 3, 0, 1], [3, 2, 1, 0], [2, 3, 1, 0], [3, 2, 0, 1]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_4d_whyx_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)


@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 1, 128, 128], [32, 32, 32, 32], [96, 96, 96, 96]])
@pytest.mark.parametrize("perm", [[0, 2, 3, 1], [0, 3, 1, 2], [1, 2, 3, 0], [2, 1, 3, 0], [2, 0, 3, 1], [1, 0, 2, 3]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        pytest.param(
            ttnn.int32,
            # marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_4d_other_permutations(device, shape, perm, dtype):
    torch.manual_seed(2005)
    values = torch.arange(96**3).repeat_interleave(96)
    # torch_tensor = values.view(96, 96, 96, 96).to(torch.int32)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape

    print("diff", torch.sum(torch_output != output_tensor).item())

    # results = find_tensor_swaps(torch_output, output_tensor)
    # print_swap_analysis(results)
    # print(find_swapped_leaves(torch_output, output_tensor))

    diff_indices = torch.where(output_tensor != torch_output)
    print("count", len(diff_indices[0]))
    # idxs = []
    last_idx = (-1, -1, -1, -1)
    for i in range(len(diff_indices[0])):
        idx = tuple(dim_idx[i].item() for dim_idx in diff_indices)
        # if last_idx[-2] == idx[-2]: continue
        val1 = torch_output[idx].item()
        val2 = output_tensor[idx].item()
        print(f"Index {idx}: {val1} != {val2}")
        last_idx = idx

    # print(torch_output)
    # print(output_tensor)
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize("shape", [[33, 1, 17, 33, 33]])
@pytest.mark.parametrize("perm", [[0, 1, 4, 2, 3], [0, 4, 1, 2, 3], [2, 4, 1, 0, 3], [4, 2, 1, 0, 3]])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
        pytest.param(
            ttnn.int32,
            marks=skip_for_wormhole_b0("possible race condition: https://github.com/tenstorrent/tt-metal/issues/22298"),
        ),
    ],
)
def test_permute_5d_wyh(device, shape, perm, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output_tensor = ttnn.permute(input_tensor, perm, pad_value=0.0)
    output_tensor = ttnn.to_torch(output_tensor)
    torch_output = torch.permute(torch_tensor, perm)
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 1.0)
