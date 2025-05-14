import ttnn
import torch

torch.manual_seed(0)
device = ttnn.open_device(device_id=0)

def test_integer_permute(device, is_32bit):
    permutation = [0, 2, 1]
    if is_32bit:
        dtype = ttnn.int32
        shape = (5, 32, 32)
        torch_input = torch.randint(0, 10, shape)
    else:
        dtype = ttnn.bfloat16
        shape = (2 * 5, 32, 32)
        torch_input = torch.randint(-10, 0, shape).bfloat16()

    torch_output = torch_input # torch.permute(torch_input, permutation)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    ttnn_output = ttnn.permute(input_tensor, permutation)
    ttnn_output = ttnn.to_torch(ttnn_output)

    if not is_32bit:
        ttnn_output = ttnn_output.bfloat16()

    for i, a in enumerate(torch_output):
        for j, row in enumerate(a):
            if not torch.all(ttnn_output[i][j] == row):
                print(i, j, ttnn_output[i][j], row)

    assert torch_output.shape == ttnn_output.shape
    assert torch.all(torch_output == ttnn_output)

try:
    test_integer_permute(device, 0)
    test_integer_permute(device, 1)
finally:
    ttnn.close_device(device)
