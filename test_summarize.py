"""Summary and runner regressions without loading model weights."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import experiment
from summarize import summarize

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_run(tmp_path):
    prompt = dict(id='fresh-01', category='control', prompt='Reply ok', expected='ok')
    write_json(tmp_path/'prompts.json', [prompt])
    write_json(tmp_path/'metadata.json', dict(model='test', revision='test', gpu=None,
               calibration=[], files={'experiment.py': hashlib.sha256(b'original').hexdigest()},
               condition_scope={'fresh-01': ['original', 'recipe']}))
    return prompt


def response(tokens=None):
    return dict(text='ok', token_ids=tokens or [1], reached_token_limit=False, seconds=0,
                activation_trace={})


def test_cli_legacy_run_without_source(tmp_path):
    prompt = make_run(tmp_path)
    write_json(tmp_path/'fresh-01.json', dict(**prompt, conditions={'original': response()}))
    result = subprocess.run([sys.executable, str(ROOT/'summarize.py'), str(tmp_path)],
                            capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary['source_provenance'] == dict(status='unavailable', missing_files=['experiment.py'])
    assert summary['submitted_source_hashes_match'] == {'experiment.py': None}
    assert summary['generated_responses'] == 1
    assert summary['completed'] is False
    assert summary['completed_prompts'] == 0
    assert summary['paired_prompts'] == 0
    assert list(csv.DictReader((tmp_path/'responses.tsv').open(), delimiter='\t'))[0]['finish_reason'] == 'unknown'


def test_source_mismatch_stays_distinct_from_missing(tmp_path):
    make_run(tmp_path)
    source = tmp_path/'source'
    source.mkdir()
    (source/'experiment.py').write_bytes(b'changed')
    result = summarize(tmp_path)
    assert result['source_provenance']['status'] == 'mismatch'
    assert result['submitted_source_hashes_match'] == {'experiment.py': False}
    (source/'experiment.py').write_bytes(b'original')
    assert summarize(tmp_path)['source_provenance']['status'] == 'verified'


def test_empty_run_writes_header_and_does_not_claim_completion(tmp_path):
    make_run(tmp_path)
    write_json(tmp_path/'complete.json', {'prompts': 1})
    result = summarize(tmp_path)
    assert result['generated_responses'] == result['paired_prompts'] == 0
    assert result['completed'] is False
    assert result['token_limit_counts'] == {}
    assert len((tmp_path/'responses.tsv').read_text().splitlines()) == 1


def test_diagnostic_partial_run_is_incomplete(tmp_path):
    prompt = make_run(tmp_path)
    metadata = json.loads((tmp_path/'metadata.json').read_text())
    metadata['condition_scope']['fresh-01'].append('L14_only')
    write_json(tmp_path/'metadata.json', metadata)
    write_json(tmp_path/'fresh-01.json', dict(**prompt, conditions={
        'original': response(), 'recipe': response([2])}))
    write_json(tmp_path/'complete.json', {'prompts': 1})
    result = summarize(tmp_path)
    assert result['paired_prompts'] == result['different_token_sequences'] == 1
    assert result['completed'] is False


def test_fresh_runner_snapshots_before_load_and_summarizes(tmp_path, monkeypatch):
    prompt_path = tmp_path/'custom.json'
    write_json(prompt_path, [dict(id='fresh-01', category='test', prompt='ok'),
                             dict(id='identity-01', category='test', prompt='who')])
    output = tmp_path/'run'

    class FakeLab:
        def __init__(self, device):
            self.device = device
            self.dtype = 'test'
            self.calibration = []
            self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9]))
            self.tok = SimpleNamespace(apply_chat_template=self.template)

        def load(self):
            metadata = json.loads((output/'metadata.json').read_text())
            assert set(metadata['files']) == {'experiment.py', 'lab.py', 'uv.lock', 'pyproject.toml', 'prompts/custom.json'}
            for name, digest in metadata['files'].items():
                assert hashlib.sha256((output/'source'/name).read_bytes()).hexdigest() == digest
            assert json.loads((output/'source/prompts/custom.json').read_text()) == json.loads(prompt_path.read_text())
            assert metadata['condition_scope'] == {'fresh-01': ['original', 'recipe']}
            assert '0 selected diagnostic prompts' in metadata['design']

        def template(self, *args, **kwargs):
            import torch
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs(input_ids=torch.tensor([[5]]))

        def generate(self, inputs, deltas, max_tokens):
            result = response([1, 9])
            result['reached_token_limit'] = True
            return result

    monkeypatch.setattr(experiment, 'Lab', FakeLab)
    monkeypatch.setattr(sys, 'argv', ['experiment.py', '--prompts', str(prompt_path), '--output', str(output),
                                    '--device', 'cpu', '--max-tokens', '2', '--limit', '1'])
    experiment.main()
    result = summarize(output)
    assert result['source_provenance']['status'] == 'verified'
    assert result['completed'] is True
    assert result['generated_responses'] == 2
    record = json.loads((output/'fresh-01.json').read_text())
    assert record['conditions']['original']['finish_reason'] == 'eos'
    assert record['conditions']['original']['truncated'] is False
    assert record['conditions']['original']['reached_token_limit'] is True


@pytest.mark.parametrize('tokens,eos,reason,truncated', [
    ([1, 9], [9], 'eos', False), ([1, 2], [9], 'length', True), ([1], [9], 'unknown', None)])
def test_completion_metadata(tokens, eos, reason, truncated):
    result = experiment.completion_metadata(response(tokens), eos, 2)
    assert result['finish_reason'] == reason
    assert result['truncated'] is truncated
