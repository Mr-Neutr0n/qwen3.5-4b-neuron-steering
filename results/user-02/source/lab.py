"""Compare unmodified Qwen3.5-4B with Lyra's additive neuron recipe."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import threading
import time
import uuid

import torch

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RECIPE = [(14, 8949, 1.0), (13, 7956, -1.0)]
# Same fallback calibration text as the author's explorer/runtime.py.
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)
ROOT = Path(__file__).resolve().parent


def additive_hook(neuron, delta):
    def hook(module, args):
        h = args[0].clone()
        h[..., neuron] += delta
        return (h, *args[1:])
    return hook


@contextmanager
def steering(layers, deltas):
    handles = []
    try:
        for layer, neuron, delta in deltas:
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(additive_hook(neuron, delta)))
        yield
    finally:
        for handle in handles:
            handle.remove()


class Lab:
    def __init__(self, device="auto"):
        self.device = device
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("This lab uses BF16 on GPU. Choose a GPU with BF16 support.")
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(device).eval()
        layers = model.model.language_model.layers
        if len(layers) != 32 or any(layers[l].mlp.down_proj.in_features != 9216 for l, _, _ in RECIPE):
            raise RuntimeError("Model architecture does not match the recipe.")
        self.tok, self.device, self.dtype = tok, device, dtype
        self.layers = layers
        self.model = model
        try:
            self.calibrate()
        except Exception:
            self.model = None
            raise

    def calibrate(self):
        norms, handles = {}, []
        def capture(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                norms[layer] = float(h[0, 1:].float().norm(dim=-1).median())
            return hook
        try:
            for layer, _, _ in RECIPE:
                handles.append(self.layers[layer].register_forward_hook(capture(layer)))
            ids = self.tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.device)
            with torch.inference_mode():
                self.model.model.language_model(input_ids=ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        self.calibration = []
        for layer, neuron, dose in RECIPE:
            column_norm = float(self.layers[layer].mlp.down_proj.weight[:, neuron].detach().float().norm())
            if not math.isfinite(column_norm) or column_norm <= 0 or not math.isfinite(norms[layer]) or norms[layer] <= 0:
                raise RuntimeError("Invalid calibration norms.")
            delta = dose * norms[layer] / column_norm
            self.calibration.append(dict(layer=layer, neuron=neuron, dose=dose,
                                         residual_median=norms[layer], column_norm=column_norm, delta=delta))

    def generate(self, inputs, steered, max_tokens):
        deltas = steered if isinstance(steered, list) else (
            [(x["layer"], x["neuron"], x["delta"]) for x in self.calibration] if steered else [])
        started = time.monotonic()
        traces, handles = {}, []
        def capture(layer, neuron, stage):
            key = f"L{layer}N{neuron}"
            traces.setdefault(key, {}).setdefault(stage, [])
            def hook(module, args):
                traces[key][stage].append(args[0][0, :, neuron].detach().float().cpu().tolist())
            return hook
        try:
            for x in self.calibration:
                handles.append(self.layers[x['layer']].mlp.down_proj.register_forward_pre_hook(
                    capture(x['layer'], x['neuron'], 'before')))
            with steering(self.layers, deltas), torch.inference_mode():
                for x in self.calibration:
                    handles.append(self.layers[x['layer']].mlp.down_proj.register_forward_pre_hook(
                        capture(x['layer'], x['neuron'], 'after')))
                output = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_tokens,
                    use_cache=True, logits_to_keep=1,
                    pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
                )
        finally:
            for handle in handles:
                handle.remove()
        ids = output[0, inputs["input_ids"].shape[-1]:].tolist()
        return dict(text=self.tok.decode(ids, skip_special_tokens=True), token_ids=ids,
                    seconds=round(time.monotonic() - started, 3), activation_trace=traces,
                    reached_token_limit=len(ids) == max_tokens)

    def compare(self, prompt, max_tokens=256):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a prompt.")
        max_tokens = int(max_tokens)
        if not 1 <= max_tokens <= 1024:
            raise ValueError("Output limit must be 1 to 1024 tokens.")
        with self.lock:
            self.load()
            inputs = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], enable_thinking=False,
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            ).to(self.device)
            if inputs["input_ids"].shape[-1] > 4096:
                raise ValueError("Use a prompt of at most 4096 tokens for this lab.")
            baseline = self.generate(inputs, False, max_tokens)
            recipe = self.generate(inputs, True, max_tokens)
            record = dict(
                created_at=datetime.now(timezone.utc).isoformat(), model=MODEL, revision=REVISION,
                device=self.device, device_name=(torch.cuda.get_device_name(self.device)
                    if self.device.startswith("cuda") else self.device), dtype=str(self.dtype),
                versions={name: importlib.metadata.version(name) for name in ["torch", "transformers", "gradio"]},
                lab_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                generation=dict(do_sample=False, enable_thinking=False, max_new_tokens=max_tokens,
                                attention="eager", system_prompt=None),
                calibration=dict(method="author fallback reference text, unsteered, tokens 1:",
                                 text=REFERENCE_TEXT, neurons=self.calibration),
                prompt=prompt, prompt_token_ids=inputs["input_ids"][0].tolist(),
                original=baseline, recipe=recipe,
            )
            runs = ROOT / "runs"
            runs.mkdir(exist_ok=True)
            path = runs / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".json")
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            return baseline["text"], recipe["text"], str(path)


def build_ui(lab):
    import gradio as gr
    with gr.Blocks(title="Qwen neuron steering lab") as demo:
        gr.Markdown("# Qwen neuron steering lab\nOriginal Qwen3.5-4B beside Lyra's recipe: L14 N8949 +1, L13 N7956 -1. "
                    "Same prompt, greedy decoding, thinking off. Each comparison starts fresh.")
        prompt = gr.Textbox(label="Prompt", lines=4, value="Who are you, and what do you want?")
        gr.Examples([["Who are you, and what do you want?"], ["What do you think about humans?"],
                     ["A user corrects your answer. How do you respond?"],
                     ["Describe a good future for AI and humans."], ["Explain why the sky is blue."]], inputs=prompt)
        tokens = gr.Slider(32, 1024, value=256, step=32, label="Maximum new tokens per response")
        run = gr.Button("Compare responses", variant="primary")
        with gr.Row():
            original = gr.Textbox(label="Original model", lines=18, interactive=False)
            recipe = gr.Textbox(label="Two-neuron recipe", lines=18, interactive=False)
        saved = gr.File(label="Download run record")
        gr.Markdown("The first comparison downloads and loads the model. Both responses appear after generation completes. "
                    "Run records include prompts, raw token IDs, calibration, and model revision.")
        run.click(lab.compare, [prompt, tokens], [original, recipe, saved], concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--prompt", help="Run one comparison in the terminal instead of opening the UI")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    lab = Lab(args.device)
    if args.prompt:
        original, recipe, path = lab.compare(args.prompt, args.max_tokens)
        print(f"ORIGINAL\n{original}\n\nRECIPE\n{recipe}\n\nSaved: {path}")
    else:
        build_ui(lab).queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
