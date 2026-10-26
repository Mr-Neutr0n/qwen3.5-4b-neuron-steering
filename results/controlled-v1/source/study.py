"""Controlled, resumable Qwen steering study. Inputs and source are frozen per run."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import shutil
import time

import torch
from lab import Lab, MODEL, REVISION, REFERENCE_TEXT

CORE = ["baseline", "noop", "recipe_0.125", "recipe_0.25", "recipe_0.5", "recipe_1",
        "L14_only", "L13_only", "reversed", "prefill_only", "decode_only", "ablation",
        "ablation_prefill", "baseline_rescue", "engineering_rescue"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_protocol(path):
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != 1:
        raise ValueError("Expected protocol schema_version 1")
    for name, digest in protocol.get("frozen_sha256", {}).items():
        if sha(path.parent / name) != digest:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    if "core_conditions" in protocol and protocol["core_conditions"] != CORE:
        raise ValueError("Core conditions differ from frozen protocol")
    prompts_path = path.parent / protocol.get("prompts", "holdout.json")
    prompts = json.loads(prompts_path.read_text())
    if isinstance(prompts, dict):
        prompts = prompts["prompts"]
    ids = [p["id"] for p in prompts]
    if not prompts or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        raise ValueError("Prompt IDs must be unique safe filenames")
    if any(not isinstance(p.get("prompt"), str) or not p["prompt"].strip() for p in prompts):
        raise ValueError("Every prompt needs nonempty prompt text")
    if not set(protocol.get("control_prompt_ids", [])) <= set(ids):
        raise ValueError("Unknown control prompt IDs")
    if not 1 <= protocol.get("max_new_tokens", 192) <= 1024:
        raise ValueError("max_new_tokens must be 1..1024")
    if not 1 <= protocol.get("batch_size", 8) <= 64:
        raise ValueError("batch_size must be 1..64")
    if protocol.get("random_controls", 12) != 12:
        raise ValueError("This protocol requires 12 controls of each kind")
    return protocol, prompts, prompts_path


def condition_names(prompt, protocol):
    names = list(CORE)
    if prompt["id"] in protocol.get("control_prompt_ids", []):
        names += [f"random_{kind}_{i:02d}" for kind in ("neuron", "residual") for i in range(12)]
    return names


def build_conditions(lab, seed):
    """Normalize each control at each layer to that layer's recipe output norm."""
    definitions = {name: {"name": name, "phase": "all", "operations": []} for name in CORE}
    for name in CORE:
        definition = definitions[name]
        if name in ("baseline", "noop"):
            continue
        if name in ("prefill_only", "ablation_prefill", "baseline_rescue"):
            definition["phase"] = "prefill"
        if name == "decode_only":
            definition["phase"] = "decode"
        for item in lab.calibration:
            layer, neuron = item["layer"], item["neuron"]
            operation = dict(layer=layer, neuron=neuron, kind="add", delta=item["delta"])
            if name.startswith("recipe_"):
                operation["delta"] *= float(name.split("_")[1])
            elif name == "L14_only" and layer != 14:
                continue
            elif name == "L13_only" and layer != 13:
                continue
            elif name == "reversed":
                operation["delta"] *= -1
            elif name.startswith("ablation") or name == "baseline_rescue":
                operation["kind"] = "zero"
            definition["operations"].append(operation)
        if name in ("baseline_rescue", "engineering_rescue"):
            definition["replace_baseline_prefill"] = True
    rng = random.Random(seed)
    for i in range(12):
        neurons, residuals = [], []
        for item in lab.calibration:
            layer = item["layer"]
            weight = lab.layers[layer].mlp.down_proj.weight.detach()
            candidates = list(range(weight.shape[1]))
            candidates.remove(item["neuron"])
            neuron = rng.choice(candidates)
            sign = rng.choice([-1, 1])
            norm = float(weight[:, neuron].float().norm())
            if norm <= 0:
                raise ValueError("Zero column norm in random control")
            neurons.append(dict(layer=layer, neuron=neuron, kind="add",
                                delta=sign * item["residual_median"] / norm,
                                intended_output_norm=item["residual_median"]))
            residuals.append(dict(layer=layer, kind="residual", vector_seed=seed + 10000 + i * 100 + layer,
                                  intended_output_norm=item["residual_median"]))
        for kind, ops in (("neuron", neurons), ("residual", residuals)):
            name = f"random_{kind}_{i:02d}"
            definitions[name] = dict(name=name, phase="all", operations=ops)
    return definitions


def residual_vector(operation, size, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(operation["vector_seed"])
    value = torch.randn(size, generator=generator, dtype=torch.float32)
    value *= operation["intended_output_norm"] / value.norm()
    return value.to(device=device, dtype=dtype)


@contextmanager
def interventions(lab, conditions, natural=None):
    """One batch row per condition. Call count, not length, identifies prefill."""
    handles, calls, captures = [], {}, {}
    first_logits = []
    vectors = {}
    for row, definition in enumerate(conditions):
        for op in definition["operations"]:
            if op["kind"] == "residual":
                weight = lab.layers[op["layer"]].mlp.down_proj.weight
                vectors[row, op["layer"]] = residual_vector(op, weight.shape[0], weight.device, weight.dtype)
    baseline_row = next((i for i, d in enumerate(conditions) if d["name"] == "baseline"), None)
    target_by_layer = {x["layer"]: x["neuron"] for x in lab.calibration}

    def before(layer):
        def hook(module, args):
            first = calls.get(layer, 0) == 0
            calls[layer] = calls.get(layer, 0) + 1
            h = args[0]
            neuron = target_by_layer[layer]
            if first:
                captures[layer] = {"before": h[..., neuron].detach().float().cpu().tolist()}
            # Baseline installs observation hooks only; noop explicitly clones.
            changed = any(d["name"] != "baseline" for d in conditions)
            out = h.clone() if changed else h
            for row, definition in enumerate(conditions):
                phase = definition["phase"]
                active = phase == "all" or (phase == "prefill" and first) or (phase == "decode" and not first)
                if active:
                    for op in definition["operations"]:
                        if op["layer"] == layer and op["kind"] != "residual":
                            if op["kind"] == "zero":
                                out[row, :, op["neuron"]] = 0
                            else:
                                out[row, :, op["neuron"]] += op["delta"]
                if first and definition.get("replace_baseline_prefill"):
                    if baseline_row is not None:
                        value = h[baseline_row, :, neuron]
                    elif natural is not None:
                        value = torch.tensor(natural[layer], device=h.device, dtype=h.dtype)
                    else:
                        raise ValueError("Baseline replacement needs natural prefill captures")
                    if value.shape != out[row, :, neuron].shape:
                        raise ValueError("Natural prefill shape mismatch")
                    out[row, :, neuron] = value
            if first:
                captures[layer]["after"] = out[..., neuron].detach().float().cpu().tolist()
            return (out, *args[1:])
        return hook

    def after(layer):
        def hook(module, args, output):
            first = calls[layer] == 1
            out = output
            for row, definition in enumerate(conditions):
                phase = definition["phase"]
                active = phase == "all" or (phase == "prefill" and first) or (phase == "decode" and not first)
                if active and (row, layer) in vectors:
                    if out is output:
                        out = output.clone()
                    out[row] += vectors[row, layer]
            return out
        return hook

    def logits_hook(module, args, output):
        if not first_logits:
            first_logits.append(output.logits[:, -1, :].detach().float().cpu())

    try:
        for layer in target_by_layer:
            projection = lab.layers[layer].mlp.down_proj
            handles.append(projection.register_forward_pre_hook(before(layer)))
            handles.append(projection.register_forward_hook(after(layer)))
        handles.append(lab.model.register_forward_hook(logits_hook))
        yield captures, first_logits
    finally:
        for handle in handles:
            handle.remove()


def generation_eos_ids(lab):
    eos = lab.model.generation_config.eos_token_id
    if eos is None:
        eos = lab.tok.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos or [])


def diagnostics(ids, tok, limit, eos_token_ids=None):
    raw_ids = list(ids)
    eos = tok.eos_token_id if eos_token_ids is None else eos_token_ids
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    end = next((i + 1 for i, value in enumerate(ids) if value in eos_ids), len(ids))
    ids = ids[:end]
    raw = tok.decode(ids, skip_special_tokens=False)
    text = tok.decode(ids, skip_special_tokens=True)
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    stopped_eos = bool(ids and ids[-1] in eos_ids)
    capped = len(ids) >= limit and not stopped_eos
    return dict(text=text, raw_text=raw, token_ids=ids, generated_tokens=len(ids),
                generated_tensor_token_ids=raw_ids,
                termination=dict(eos=stopped_eos, reached_token_limit=capped,
                                 length_at_limit=len(ids) >= limit,
                                 reason="eos" if stopped_eos else "token_limit" if capped else "other",
                                 eos_token_ids=sorted(eos_ids),
                                 padding_tokens_removed=len(raw_ids) - len(ids)),
                diagnostics=dict(repeated_4gram_fraction=1 - len(set(grams)) / len(grams) if grams else 0,
                                 role_markers={marker: raw.count(marker) for marker in
                                               ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "\nuser\n", "\nassistant\n")}))


def generate_batch(lab, inputs, definitions, max_tokens, natural=None, baseline_logits=None, baseline_tokens=None):
    batch = {key: value.repeat(len(definitions), 1) for key, value in inputs.items()}
    eos_ids = generation_eos_ids(lab)
    pad_id = lab.tok.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0] if eos_ids else None
    start = time.monotonic()
    with interventions(lab, definitions, natural) as (captures, logits), torch.inference_mode():
        output = lab.model.generate(**batch, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                                    logits_to_keep=1, pad_token_id=pad_id, eos_token_id=eos_ids or None)
    records = {}
    baseline_row = next((i for i, d in enumerate(definitions) if d["name"] == "baseline"), None)
    matched = None
    if baseline_row is not None:
        matched = diagnostics(output[baseline_row, inputs["input_ids"].shape[-1]:].tolist(), lab.tok, max_tokens, eos_ids)
        baseline_logits = logits[0][baseline_row]
    for row, definition in enumerate(definitions):
        record = diagnostics(output[row, inputs["input_ids"].shape[-1]:].tolist(), lab.tok, max_tokens, eos_ids)
        record["batch_seconds"] = round(time.monotonic() - start, 3)
        record["prompt_activations"] = {f"L{layer}": {stage: values[row] for stage, values in stages.items()}
                                         for layer, stages in captures.items()}
        if matched is not None:
            record["matched_baseline"] = dict(token_ids=matched["token_ids"], text=matched["text"],
                                               raw_text=matched["raw_text"], termination=matched["termination"],
                                               batch_size=len(definitions),
                                               matches_singleton=matched["token_ids"] == baseline_tokens
                                               if baseline_tokens is not None else None)
            record["matches_matched_baseline"] = record["token_ids"] == matched["token_ids"]
        if definition.get("replace_baseline_prefill"):
            record["replacement_source"] = "same_batch_baseline" if baseline_row is not None else "supplied_natural"
        record["kl_reference"] = "same_batch_baseline" if matched is not None else "supplied_baseline"
        if baseline_logits is not None:
            logp = baseline_logits.log_softmax(-1)
            logq = logits[0][row].log_softmax(-1)
            record["first_token_kl_baseline"] = float((logp.exp() * (logp - logq)).sum().clamp_min(0))
        records[definition["name"]] = record
    return records, captures, logits[0]


def source_paths(protocol_path, prompts_path):
    root = Path(__file__).resolve().parent
    sources = {"study.py": Path(__file__), "lab.py": root / "lab.py", "protocol.json": Path(protocol_path),
               "holdout.json": prompts_path}
    for name in ("pyproject.toml", "uv.lock", "test_study.py"):
        if (root / name).exists():
            sources[name] = root / name
    for name in ("rubric.json", "rubric.md"):
        rubric = Path(protocol_path).resolve().parent / name
        if rubric.exists():
            sources[name] = rubric
    return sources


def run(args):
    protocol, prompts, prompts_path = read_protocol(args.protocol)
    seed = protocol.get("seed", 20261026)
    plan = {p["id"]: condition_names(p, protocol) for p in prompts}
    if args.dry_run:
        print(json.dumps(dict(prompt_count=len(prompts), response_count=sum(map(len, plan.values())),
                              max_new_tokens=protocol.get("max_new_tokens", 192), plan=plan), indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_paths(args.protocol, prompts_path)
    hashes = {name: sha(path) for name, path in sources.items()}
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_sha256"] != hashes:
            raise ValueError("Run inputs/source changed. Use a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Output directory contains files but no run metadata")
        (output / "source").mkdir()
        for name, path in sources.items():
            shutil.copyfile(path, output / "source" / name)
        metadata = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                        source_sha256=hashes, model=MODEL, revision=REVISION, protocol=protocol,
                        condition_order_seed=seed, plan=plan)
        atomic_json(metadata_path, metadata)
    complete_path = output / "complete.json"
    if complete_path.exists():
        completed = json.loads(complete_path.read_text())
        expected = completed.get("records", {})
        if (completed.get("source_sha256") != hashes or set(expected) != {p["id"] for p in prompts}
                or any(not (output / (key + ".json")).exists()
                       or sha(output / (key + ".json")) != digest for key, digest in expected.items())):
            raise ValueError("Completed run integrity check failed")
        print("Run already complete; source and record hashes verified", flush=True)
        return
    torch.manual_seed(seed)
    lab = Lab(args.device)
    lab.load()
    definitions = build_conditions(lab, seed)
    runtime = dict(device=str(lab.device), dtype=str(lab.dtype),
                   device_name=torch.cuda.get_device_name(lab.device) if str(lab.device).startswith("cuda") else str(lab.device),
                   versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   attention="eager", do_sample=False, enable_thinking=False, system_prompt=None,
                   eos_token_ids=generation_eos_ids(lab), pad_token_id=lab.tok.pad_token_id)
    if "runtime" in metadata and (metadata["runtime"] != runtime or metadata["conditions"] != definitions
                                  or metadata["calibration"]["neurons"] != lab.calibration):
        raise ValueError("Runtime/calibration changed. Use a new output directory")
    metadata.update(runtime=runtime, conditions=definitions,
                    calibration=dict(text=REFERENCE_TEXT, neurons=lab.calibration),
                    interpretation={"baseline_rescue": "A prefill ablation/replacement sanity check; not evidence of natural mediation.",
                                    "engineering_rescue": "Same-batch baseline prefill target replacement under the recipe; decode stays steered.",
                                    "kl": "KL(same-batch baseline || condition), same fixed prompt, first next-token distribution.",
                                    "norms": "Control norms are matched in FP32 before BF16 rounding at each layer independently."})
    atomic_json(metadata_path, metadata)
    max_tokens = protocol.get("max_new_tokens", 192)
    for prompt in prompts:
        path = output / (prompt["id"] + ".json")
        record = json.loads(path.read_text()) if path.exists() else dict(schema_version=1, prompt=prompt, conditions={})
        if record["prompt"] != prompt:
            raise ValueError("Saved prompt differs")
        inputs = lab.tok.apply_chat_template([{"role": "user", "content": prompt["prompt"]}],
                                            enable_thinking=False, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(lab.device)
        if inputs["input_ids"].shape[-1] > 4096:
            raise ValueError("Prompt exceeds 4096 tokens")
        record["prompt_token_ids"] = inputs["input_ids"][0].tolist()
        baseline, captures, logits = generate_batch(lab, inputs, [definitions["baseline"]], max_tokens)
        if "baseline" in record["conditions"] and baseline["baseline"]["token_ids"] != record["conditions"]["baseline"]["token_ids"]:
            raise RuntimeError("Baseline differs on resume")
        record["conditions"].update(baseline)
        record["conditions"]["baseline"]["first_token_kl_baseline"] = 0.0
        natural = {layer: stages["before"][0] for layer, stages in captures.items()}
        names = [x for x in plan[prompt["id"]] if x != "baseline"]
        prompt_seed = seed + int(hashlib.sha256(prompt["id"].encode()).hexdigest()[:8], 16)
        random.Random(prompt_seed).shuffle(names)
        record["condition_order"] = ["baseline"] + names
        atomic_json(path, record)
        remaining = [name for name in names if name not in record["conditions"]]
        size = protocol.get("batch_size", 8)
        for offset in range(0, len(remaining), size):
            chunk = remaining[offset:offset + size]
            values, _, _ = generate_batch(lab, inputs, [definitions["baseline"]] + [definitions[x] for x in chunk],
                                           max_tokens, natural, logits[0], baseline["baseline"]["token_ids"])
            values.pop("baseline")
            record["conditions"].update(values)
            atomic_json(path, record)
            print(f"{prompt['id']}: {len(record['conditions'])}/{len(plan[prompt['id']])}", flush=True)
        restored, _, _ = generate_batch(lab, inputs, [definitions["baseline"]], max_tokens)
        record["restoration"] = dict(token_ids=restored["baseline"]["token_ids"],
                                     matches=restored["baseline"]["token_ids"] == baseline["baseline"]["token_ids"])
        atomic_json(path, record)
        if not record["restoration"]["matches"]:
            raise RuntimeError("Baseline restoration failed")
    atomic_json(output / "complete.json", dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                               prompt_count=len(prompts), source_sha256=hashes,
                                               records={p["id"]: sha(output / (p["id"] + ".json")) for p in prompts}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="study/protocol.json")
    parser.add_argument("--output", default="results/controlled-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
