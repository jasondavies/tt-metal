# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.sentence_bert.common import load_torch_model
from models.demos.sentence_bert.reference.sentence_bert import BertEmbeddings
from models.demos.sentence_bert.ttnn.common import custom_preprocessor, preprocess_inputs
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_embeddings import TtnnSentenceBertEmbeddings
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [8, 384]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_Embeddings(device, inputs, model_location_generator):
    # transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).embeddings.eval()
    # config = transformers.BertConfig.from_pretrained(inputs[0])

    target_prefix = f"embeddings."
    config = transformers.BertConfig.from_pretrained(inputs[0])
    input_ids = torch.randint(low=0, high=config.vocab_size - 1, size=inputs[1], dtype=torch.int64)
    attention_mask = torch.randint(low=0, high=inputs[1][1], size=inputs[1], dtype=torch.int64)
    token_type_ids = torch.zeros(inputs[1], dtype=torch.int64)
    position_ids = torch.arange(0, inputs[1][1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertEmbeddings(config)
    reference_module = load_torch_model(
        reference_module, target_prefix=target_prefix, model_location_generator=model_location_generator
    )
    reference_out = reference_module(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnSentenceBertEmbeddings(parameters, config)
    ttnn_input_ids, ttnn_token_type_ids, ttnn_position_ids, _, _ = preprocess_inputs(
        input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, device=device
    )
    ttnn_out = ttnn_module(
        input_ids=ttnn_input_ids, token_type_ids=ttnn_token_type_ids, position_ids=ttnn_position_ids, device=device
    )
    ttnn_out = ttnn.to_torch(ttnn_out).squeeze(dim=1)
    assert_with_pcc(reference_out, ttnn_out, 0.99)
