# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.vovnet.tt.osa_stage import TtOsaStage
from models.experimental.vovnet.tt.model_preprocessing import custom_preprocessor
from models.experimental.vovnet.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_osa_stage_inference(device, reset_seeds, model_location_generator):
    STAGE_INDEX = 3

    base_address = f"stages.{STAGE_INDEX}"
    model = load_torch_model(model_location_generator)
    parameters = custom_preprocessor(device, model.state_dict())
    torch_model = model.stages[STAGE_INDEX]

    downsample = False
    if STAGE_INDEX > 0:
        downsample = True
    tt_model = TtOsaStage(
        base_address=base_address,
        parameters=parameters,
        device=device,
        downsample=downsample,
    )

    input = torch.randn(1, 768, 14, 14)
    model_output = torch_model(input)

    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(model_output, tt_output_torch, 0.99)
