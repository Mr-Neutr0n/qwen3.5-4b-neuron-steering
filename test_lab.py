from types import SimpleNamespace

import pytest
import torch

from lab import Lab, build_ui, steering


def test_push_is_normalized_column_at_every_token_and_restores_baseline():
    projection = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))]
    inputs = torch.randn(2, 4, 3)
    saved_inputs = inputs.clone()
    baseline = projection(inputs)
    median_residual = 7.0
    column = projection.weight[:, 1]
    delta = -median_residual / column.norm().item()
    with steering(layers, [(0, 1, delta)]):
        difference = projection(inputs) - baseline
        expected = (column * delta).expand_as(difference)
        torch.testing.assert_close(difference, expected)
        torch.testing.assert_close(difference.norm(dim=-1), torch.full((2, 4), median_residual))
        # Decode calls have one token instead of the prompt's four.
        torch.testing.assert_close(projection(inputs[:, :1]) - baseline[:, :1], expected[:, :1])
    torch.testing.assert_close(inputs, saved_inputs)
    torch.testing.assert_close(projection(inputs), baseline)
    assert not projection._forward_pre_hooks


def test_hooks_removed_on_failure_including_partial_install():
    projection = torch.nn.Linear(3, 2)
    layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))]
    with pytest.raises(RuntimeError):
        with steering(layers, [(0, 1, 2)]):
            raise RuntimeError("generation failed")
    assert not projection._forward_pre_hooks
    with pytest.raises(IndexError):
        with steering(layers, [(0, 1, 2), (5, 1, 3)]):
            pass
    assert not projection._forward_pre_hooks


def test_ui_build_does_not_load_model():
    lab = Lab("cpu")
    ui = build_ui(lab)
    assert lab.model is None
    assert ui.get_config_file()["components"]


def test_empty_prompt_rejected_before_loading():
    lab = Lab("cpu")
    with pytest.raises(ValueError, match="Enter a prompt"):
        lab.compare("  ")
    assert lab.model is None


def test_transformers_calibration_and_generation_with_tiny_random_qwen(monkeypatch):
    import lab as module
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    monkeypatch.setattr(module, "RECIPE", [(1, 5, 1.0), (0, 3, -1.0)])
    config = Qwen3_5Config(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         layer_types=["linear_attention", "full_attention"],
                         pad_token_id=0, eos_token_id=1),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, patch_size=2, spatial_merge_size=2,
                           temporal_patch_size=2, deepstack_visual_indexes=[]),
    )
    model = Qwen3_5ForConditionalGeneration(config).eval()
    class TinyTokenizer:
        pad_token_id, eos_token_id = 0, 1
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[2, 3, 4, 5]]))
        def decode(self, ids, **kwargs):
            return str(ids)
    lab = Lab("cpu")
    lab.model, lab.tok = model, TinyTokenizer()
    lab.layers = model.model.language_model.layers
    lab.calibrate()
    inputs = dict(input_ids=torch.tensor([[2, 3, 4, 5]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
    baseline = lab.generate(inputs, False, 3)
    pushed = lab.generate(inputs, True, 3)
    assert pushed["token_ids"]
    for neuron in lab.calibration:
        key = f"L{neuron['layer']}N{neuron['neuron']}"
        before = torch.tensor(pushed['activation_trace'][key]['before'][0])
        after = torch.tensor(pushed['activation_trace'][key]['after'][0])
        torch.testing.assert_close(after - before, torch.full_like(before, neuron['delta']))
        assert baseline['activation_trace'][key]['before'] == baseline['activation_trace'][key]['after']
    assert lab.generate(inputs, False, 3)["token_ids"] == baseline["token_ids"]
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in lab.layers)
