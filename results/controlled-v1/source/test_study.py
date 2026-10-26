"""CPU checks for controlled interventions and saved measurements."""
from types import SimpleNamespace
import pytest
import torch
from study import build_conditions, generate_batch, interventions, residual_vector, read_protocol

@pytest.fixture(scope="module")
def lab():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.manual_seed(17)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         layer_types=["linear_attention", "full_attention"], pad_token_id=0, eos_token_id=1),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, patch_size=2, spatial_merge_size=2,
                           temporal_patch_size=2, deepstack_visual_indexes=[]))
    model = Qwen3_5ForConditionalGeneration(config).eval()
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids))
    result = SimpleNamespace(model=model, layers=model.model.language_model.layers, tok=Tokenizer(), calibration=[])
    for layer, neuron, sign in ((1, 5, 1), (0, 3, -1)):
        norm = float(result.layers[layer].mlp.down_proj.weight[:, neuron].detach().norm())
        result.calibration.append(dict(layer=layer, neuron=neuron, delta=sign * 2 / norm,
                                       residual_median=2., column_norm=norm))
    return result


def test_batched_controls_have_same_induced_norm(lab):
    definitions = build_conditions(lab, 71)
    assert definitions == build_conditions(lab, 71)
    for kind in ("neuron", "residual"):
        selected = [definitions[f"random_{kind}_{i:02d}"] for i in range(12)]
        for layer, neuron in ((1, 5), (0, 3)):
            projection = lab.layers[layer].mlp.down_proj
            h = torch.randn(12, 3, projection.in_features)
            baseline = projection(h)
            with interventions(lab, selected):
                pushed = projection(h)
            torch.testing.assert_close((pushed - baseline).norm(dim=-1), torch.full((12, 3), 2.), rtol=2e-5, atol=2e-5)
            for definition in selected:
                operation = next(x for x in definition["operations"] if x["layer"] == layer)
                if kind == "neuron":
                    assert operation["neuron"] != neuron
                else:
                    vector = residual_vector(operation, projection.out_features, "cpu", torch.float32)
                    torch.testing.assert_close(vector.norm(), torch.tensor(2.))


def test_prefill_decode_gating_uses_call_order_even_one_token_prompt(lab):
    definitions = build_conditions(lab, 71)
    projection = lab.layers[1].mlp.down_proj
    h = torch.randn(2, 1, projection.in_features)
    baseline = projection(h)
    with interventions(lab, [definitions["prefill_only"], definitions["decode_only"]]):
        first = projection(h)
        second = projection(h)
    torch.testing.assert_close(first[1], baseline[1])
    torch.testing.assert_close(second[0], baseline[0])
    assert not torch.allclose(first[0], baseline[0])
    assert not torch.allclose(second[1], baseline[1])


def test_cleanup_after_hook_error_restores_inputs_weights_and_output(lab):
    projection = lab.layers[1].mlp.down_proj
    definitions = build_conditions(lab, 71)
    h = torch.randn(1, 3, projection.in_features)
    original, weight, expected = h.clone(), projection.weight.detach().clone(), projection(h)
    with pytest.raises(ValueError, match="shape mismatch"):
        with interventions(lab, [definitions["engineering_rescue"]], natural={1: [0.]}):
            projection(h)
    assert not projection._forward_pre_hooks and not projection._forward_hooks
    assert not lab.model._forward_hooks
    torch.testing.assert_close(h, original)
    torch.testing.assert_close(projection.weight, weight)
    torch.testing.assert_close(projection(h), expected)


def test_natural_capture_ablation_and_replacement_semantics(lab):
    definitions = build_conditions(lab, 71)
    projection = lab.layers[1].mlp.down_proj
    h = torch.randn(1, 3, projection.in_features)
    baseline = projection(h)
    with interventions(lab, [definitions["baseline"]]) as (captures, _):
        torch.testing.assert_close(projection(h), baseline)
    natural = {1: captures[1]["before"][0]}
    with interventions(lab, [definitions["ablation_prefill"]]) as (ablated, _):
        zeroed = projection(h)
    assert ablated[1]["after"][0] == [0., 0., 0.]
    assert not torch.allclose(zeroed, baseline)
    with interventions(lab, [definitions["baseline_rescue"]], natural) as (rescued, _):
        torch.testing.assert_close(projection(h), baseline)
    assert rescued[1]["after"] == captures[1]["before"]
    with interventions(lab, [definitions["engineering_rescue"]], natural):
        torch.testing.assert_close(projection(h), baseline)
        assert not torch.allclose(projection(h[:, :1]), baseline[:, :1])


def test_tiny_qwen_generation_batch_capture_kl_and_restoration(lab):
    definitions = build_conditions(lab, 71)
    inputs = dict(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    baseline, captures, logits = generate_batch(lab, inputs, [definitions["baseline"]], 3)
    natural = {layer: stages["before"][0] for layer, stages in captures.items()}
    names = ["baseline", "noop", "recipe_1", "ablation_prefill", "baseline_rescue", "random_residual_00"]
    records, _, _ = generate_batch(lab, inputs, [definitions[name] for name in names], 3, natural, logits[0], baseline["baseline"]["token_ids"])
    assert records["baseline_rescue"]["token_ids"] == baseline["baseline"]["token_ids"]
    assert records["noop"]["token_ids"] == baseline["baseline"]["token_ids"]
    assert records["baseline_rescue"]["replacement_source"] == "same_batch_baseline"
    for name in names:
        assert records[name]["matched_baseline"]["token_ids"] == records["baseline"]["token_ids"]
        assert records[name]["matched_baseline"]["batch_size"] == len(names)
        assert records[name]["matched_baseline"]["matches_singleton"] is True
        assert records[name]["kl_reference"] == "same_batch_baseline"
    assert records["noop"]["matches_matched_baseline"]
    assert records["baseline_rescue"]["matches_matched_baseline"]
    assert records["baseline_rescue"]["first_token_kl_baseline"] < 1e-6
    assert records["noop"]["first_token_kl_baseline"] < 1e-6
    assert records["recipe_1"]["first_token_kl_baseline"] > 0
    assert records["ablation_prefill"]["first_token_kl_baseline"] > 0
    restored, _, _ = generate_batch(lab, inputs, [definitions["baseline"]], 3)
    assert restored["baseline"]["token_ids"] == baseline["baseline"]["token_ids"]
    assert all(not layer.mlp.down_proj._forward_pre_hooks and not layer.mlp.down_proj._forward_hooks for layer in lab.layers)


def test_protocol_rejects_unsafe_record_names(tmp_path):
    import json
    (tmp_path / "holdout.json").write_text(json.dumps([dict(id="../escape", prompt="hello")]))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(dict(schema_version=1, prompts="holdout.json")))
    with pytest.raises(ValueError, match="safe filenames"):
        read_protocol(path)


def test_replacement_uses_same_batch_natural_values_instead_of_singleton(lab):
    definitions = build_conditions(lab, 71)
    projection = lab.layers[1].mlp.down_proj
    h = torch.randn(2, 3, projection.in_features)
    h[1] = h[0]
    # Deliberately wrong singleton values prove the observed baseline row wins.
    with interventions(lab, [definitions["baseline"], definitions["baseline_rescue"]], {1: [99., 99., 99.]}) as (captures, _):
        output = projection(h)
    torch.testing.assert_close(output[0], output[1])
    assert captures[1]["after"][1] == captures[1]["before"][0]


def test_generation_config_alternate_eos_trims_padding_and_eos_at_cap(lab):
    from study import diagnostics
    class Model(torch.nn.Module):
        generation_config = SimpleNamespace(eos_token_id=[1, 9])
        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 1, 32))
        def generate(self, input_ids, **kwargs):
            assert kwargs["eos_token_id"] == [1, 9]
            assert kwargs["pad_token_id"] == 0
            self(input_ids)
            return torch.tensor([[2, 3, 4, 7, 9, 0, 0], [2, 3, 4, 8, 8, 8, 9]])
    stub = SimpleNamespace(model=Model(), tok=lab.tok, layers=lab.layers, calibration=lab.calibration)
    definitions = build_conditions(stub, 71)
    inputs = dict(input_ids=torch.tensor([[2, 3, 4]]))
    records, _, _ = generate_batch(stub, inputs, [definitions["baseline"], definitions["noop"]], 4)
    early, at_limit = records["baseline"], records["noop"]
    assert early["token_ids"] == [7, 9]
    assert early["generated_tensor_token_ids"] == [7, 9, 0, 0]
    assert early["termination"]["padding_tokens_removed"] == 2
    assert early["termination"]["eos"] and not early["termination"]["reached_token_limit"]
    assert at_limit["termination"]["eos"] and at_limit["termination"]["length_at_limit"]
    assert at_limit["termination"]["reason"] == "eos"
    assert not at_limit["termination"]["reached_token_limit"]
    assert at_limit["matched_baseline"]["termination"] == early["termination"]
    truncated = diagnostics([7, 7, 7, 7], lab.tok, 4, [1, 9])
    assert truncated["termination"]["reason"] == "token_limit"
    assert truncated["termination"]["reached_token_limit"] and not truncated["termination"]["eos"]
