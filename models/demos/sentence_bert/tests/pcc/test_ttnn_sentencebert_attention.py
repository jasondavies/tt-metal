# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.sentence_bert.common import load_torch_model
from models.demos.sentence_bert.reference.sentence_bert import BertAttention
from models.demos.sentence_bert.ttnn.common import custom_preprocessor
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_attention import TtnnSentenceBertAttention
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [8, 384, 768], [8, 1, 1, 384]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_attention(device, inputs, model_location_generator):
    target_prefix = f"encoder.layer.{0}.attention."

    config = transformers.BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertAttention(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix=target_prefix, model_location_generator=model_location_generator
    )
    reference_out = reference_module(hidden_states, attention_mask)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnSentenceBertAttention(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    sharded_input = ttnn.to_memory_config(
        ttnn_hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_hidden_states.shape,
            core_grid=ttnn.CoreGrid(y=8, x=6),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ttnn_attention_mask = ttnn.from_torch(attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(
        sharded_input,
        ttnn_attention_mask,
        device=device,
    )
    ttnn_out = ttnn.to_torch(ttnn_out).squeeze(dim=1)
    assert_with_pcc(reference_out[0], ttnn_out, 0.99)
