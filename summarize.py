"""Build a complete local response table and check deterministic controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def check(prompt, response):
    text = response['text'].strip()
    if 'expected' in prompt:
        return text == prompt['expected']
    if 'expected_object' in prompt:
        try:
            return json.loads(text) == prompt['expected_object']
        except ValueError:
            return False
    return None


RESPONSE_FIELDS = ['id', 'category', 'prompt', 'condition', 'response', 'tokens',
                   'reached_token_limit', 'finish_reason', 'truncated', 'exact_control_pass', 'seconds']


def source_provenance(directory, files):
    checks = {}
    for name, digest in files.items():
        path = directory/'source'/name
        checks[name] = (hashlib.sha256(path.read_bytes()).hexdigest() == digest
                        if path.is_file() else None)
    status = ('mismatch' if False in checks.values() else
              'unavailable' if not checks or None in checks.values() else 'verified')
    return dict(status=status, missing_files=[name for name, match in checks.items() if match is None]), checks


def summarize(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    prompts = json.loads((directory/'prompts.json').read_text())
    rows, checks, activation_checks = [], [], []
    for prompt in prompts:
        path = directory/(prompt['id']+'.json')
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        for condition, response in record['conditions'].items():
            requested={(l,n):d for l,n,d in response.get('interventions',[])}
            for neuron in metadata['calibration']:
                key=f"L{neuron['layer']}N{neuron['neuron']}"
                trace=response['activation_trace'][key]
                before=trace['before'][0]
                after=trace['after'][0]
                differences=[b-a for a,b in zip(before,after)]
                activation_checks.append(dict(id=prompt['id'],condition=condition,neuron=key,
                    requested_delta=requested.get((neuron['layer'],neuron['neuron']),0),
                    observed_delta_min=min(differences),observed_delta_max=max(differences),
                    observed_delta_mean=statistics.mean(differences),
                    before_min=min(before),before_max=max(before),before_mean=statistics.mean(before)))
            verdict = check(prompt, response)
            rows.append(dict(id=prompt['id'], category=prompt['category'], prompt=prompt['prompt'],
                             condition=condition, response=response['text'],
                             tokens=len(response['token_ids']), reached_token_limit=response['reached_token_limit'],
                             finish_reason=response.get('finish_reason', 'unknown'), truncated=response.get('truncated'),
                             exact_control_pass=verdict, seconds=response['seconds']))
            if verdict is not None:
                checks.append(dict(id=prompt['id'],condition=condition,passed=verdict))
    changed, paired = 0, 0
    completed_prompt_ids = []
    for prompt in prompts:
        path=directory/(prompt['id']+'.json')
        if not path.exists():
            continue
        record=json.loads(path.read_text())['conditions']
        expected = metadata.get('condition_scope', {}).get(prompt['id'], ['original', 'recipe'])
        if set(expected) <= record.keys():
            completed_prompt_ids.append(prompt['id'])
        if 'original' in record and 'recipe' in record:
            paired+=1
            changed+=record['original']['token_ids'] != record['recipe']['token_ids']
    summary = dict(model=metadata['model'],revision=metadata['revision'],gpu=metadata['gpu'],
                   completed=(bool(prompts) and (directory/'complete.json').exists() and len(completed_prompt_ids) == len(prompts)),
                   completed_prompts=len(completed_prompt_ids), expected_prompts=len(prompts), paired_prompts=paired,
                   generated_responses=len(rows),different_token_sequences=changed,
                   token_limit_counts={c:sum(r['reached_token_limit'] for r in rows if r['condition']==c)
                                       for c in sorted({r['condition'] for r in rows})},
                   exact_controls=checks,interpretation='Different text does not establish misalignment. Exact checks cover only the listed small tasks. The prompt suite is exploratory and not a representative sample.',
                   calibration=metadata['calibration'])
    if (directory/'restoration.json').exists():
        summary['restored_original_identical_tokens']=json.loads((directory/'restoration.json').read_text())['identical_tokens']
    provenance, matches = source_provenance(directory, metadata.get('files', {}))
    summary['source_provenance'] = provenance
    summary['submitted_source_hashes_match'] = matches
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (directory/'activation-checks.json').write_text(json.dumps(activation_checks,indent=2)+'\n')
    with (directory/'responses.tsv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=RESPONSE_FIELDS,delimiter='\t')
        writer.writeheader();writer.writerows(rows)
    manifest={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.rglob('*'))
              if p.is_file() and p.name != 'sha256.json'}
    (directory/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.directory),indent=2))
