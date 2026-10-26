"""Run a fixed prompt suite, save every response, and measure activation changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch
from lab import Lab, MODEL, REVISION, REFERENCE_TEXT

ROOT = Path(__file__).resolve().parent
DIAGNOSTICS = {'identity-01', 'honesty-01', 'oversight-01', 'social-01', 'boundary-01', 'control-01'}


def conditions(lab, diagnostic):
    full = [(x['layer'], x['neuron'], x['delta']) for x in lab.calibration]
    result = [('original', []), ('recipe', full)]
    if diagnostic:
        result += [('L14_only', [full[0]]), ('L13_only', [full[1]])]
        for dose in (0.5, -1.0):
            result.append((f'recipe_x{dose:g}', [(l, n, d*dose) for l, n, d in full]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prompts', type=Path, default=ROOT/'prompts.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=192)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only', nargs='*', help='Prompt IDs to run')
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if args.only:
        missing = set(args.only) - {p['id'] for p in prompts}
        if missing:
            raise ValueError(f'Unknown prompt IDs: {missing}')
        prompts = [p for p in prompts if p['id'] in args.only]
    if args.limit:
        prompts = prompts[:args.limit]
    assert prompts and len({p['id'] for p in prompts}) == len(prompts)
    args.output.mkdir(parents=True, exist_ok=False)
    lab = Lab(args.device)
    lab.load()
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), model=MODEL, revision=REVISION,
        device=lab.device, gpu=torch.cuda.get_device_name(), dtype=str(lab.dtype),
        versions={n:importlib.metadata.version(n) for n in ['torch','transformers','gradio']},
        calibration=lab.calibration, reference_text=REFERENCE_TEXT,
        generation=dict(do_sample=False, enable_thinking=False, max_new_tokens=args.max_tokens,
                        attention='eager', system_prompt=None),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'lab.py',args.prompts,ROOT/'uv.lock']},
        trace_format='Each neuron has before/after arrays per forward call. First call covers prompt_token_ids. Later calls each cover the preceding generated token. Final generated token is not forwarded. Compare prefill on matched tokens. Generated continuations can diverge.',
        design='All prompts: original and recipe. Six fixed diagnostic prompts: each neuron alone, half recipe, and reversed recipe. No random-neuron control. No behavioral rate or general alignment claim is inferred.')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/'prompts.json').write_text(json.dumps(prompts,indent=2)+'\n')
    for index, prompt in enumerate(prompts):
        inputs = lab.tok.apply_chat_template([{'role':'user','content':prompt['prompt']}],
            enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(lab.device)
        if inputs['input_ids'].shape[-1] > 4096:
            raise ValueError('Prompt too long')
        record = dict(**prompt, prompt_token_ids=inputs['input_ids'][0].tolist(), conditions={})
        for name, deltas in conditions(lab, prompt['id'] in DIAGNOSTICS):
            response = lab.generate(inputs, deltas, args.max_tokens)
            response['interventions'] = deltas
            record['conditions'][name] = response
            (args.output/(prompt['id']+'.json')).write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(dict(prompt=prompt['id'],condition=name,seconds=response['seconds'],
                                  tokens=len(response['token_ids']),text=response['text'][:180])),flush=True)
        # Run the baseline again after all interventions on the first diagnostic prompt.
        if index == 0:
            restored = lab.generate(inputs, [], args.max_tokens)
            stable = restored['token_ids'] == record['conditions']['original']['token_ids']
            (args.output/'restoration.json').write_text(json.dumps(dict(prompt=prompt['id'],identical_tokens=stable,response=restored),indent=2)+'\n')
            if not stable:
                raise RuntimeError('Original generation changed after hooks were removed')
    (args.output/'complete.json').write_text(json.dumps({'prompts':len(prompts),'completed_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')


if __name__ == '__main__':
    main()
