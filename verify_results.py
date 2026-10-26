"""Verify the published evidence without loading or downloading model weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    total_prompts = total_responses = restorations = 0
    identifiers = set()
    for run in ['trial-01', 'user-01', 'user-02']:
        directory = ROOT / 'results' / run
        prompts = json.loads((directory / 'prompts.json').read_text())
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert json.loads((directory / 'complete.json').read_text())['prompts'] == len(prompts)
        assert metadata['revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        for name, digest in metadata['files'].items():
            assert hashlib.sha256((directory / 'source' / name).read_bytes()).hexdigest() == digest
        for name, digest in json.loads((directory / 'sha256.json').read_text()).items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for prompt in prompts:
            assert prompt['id'] not in identifiers
            identifiers.add(prompt['id'])
            record = json.loads((directory / (prompt['id'] + '.json')).read_text())
            assert record['prompt'] == prompt['prompt']
            assert {'original', 'recipe'} <= record['conditions'].keys()
            for response in record['conditions'].values():
                assert 0 < len(response['token_ids']) <= metadata['generation']['max_new_tokens']
                for trace in response['activation_trace'].values():
                    assert len(trace['before'][0]) == len(record['prompt_token_ids'])
                    assert list(map(len, trace['before'])) == list(map(len, trace['after']))
            total_responses += len(record['conditions'])
        restored = json.loads((directory / 'restoration.json').read_text())
        first = json.loads((directory / (restored['prompt'] + '.json')).read_text())
        assert restored['identical_tokens']
        assert restored['response']['token_ids'] == first['conditions']['original']['token_ids']
        total_prompts += len(prompts)
        restorations += 1
    assert (total_prompts, total_responses, restorations) == (50, 124, 3)
    print('Verified 50 prompts, 124 responses, 3 identical baseline restorations, and source/artifact hashes.')


if __name__ == '__main__':
    main()
