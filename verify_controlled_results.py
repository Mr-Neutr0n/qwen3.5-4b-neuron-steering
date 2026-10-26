"""Verify recorded controlled-study outputs and blinded ratings without inference."""
import hashlib
import json
from pathlib import Path
from study_analysis import AXES, BINARY, records

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())

def main():
    directory=ROOT/'results/controlled-v1'
    meta,rs=records(directory)
    assert len(rs)==96 and sum(len(r['conditions']) for r in rs)==2016
    assert len({r['prompt']['family'] for r in rs})==24
    assert all(r['restoration']['matches'] for r in rs)
    assert all(r['conditions']['noop']['matches_matched_baseline'] for r in rs)
    byid={r['prompt']['id']:r for r in rs}
    key=read(directory/'blind-key.json');packet=read(directory/'blind/packet.json')
    assert len(key)==len(packet)==800
    packets={r['response_id']:r for r in packet};assert len(packets)==800 and packets.keys()==key.keys()
    for rid,item in key.items():
        assert rid==hashlib.sha256(f"{meta['protocol']['seed']}|{item['prompt_id']}|{item['condition']}".encode()).hexdigest()[:16]
        r=byid[item['prompt_id']];condition=item['condition']
        source=r['conditions']['recipe_1']['matched_baseline'] if condition=='baseline' else r['conditions'][condition]
        assert packets[rid]['prompt']==r['prompt']['prompt'] and packets[rid]['response']==source['text']
        assert set(packets[rid])=={'response_id','prompt','response'}
    ratings=[]
    for filename in ['reviews-a.json','reviews-b.json']:
        rows=read(directory/filename);index={r['response_id']:r for r in rows}
        assert len(rows)==len(index)==800 and index.keys()==key.keys()
        for rid,row in index.items():
            for axis in AXES:assert type(row[axis]) is int and 0<=row[axis]<=(1 if axis in BINARY else 2)
            for axis in ['harmful_endorsement','actionable_harmful_assistance','contradiction','fabricated_conversation']:
                if row[axis]:
                    evidence=row['evidence'][axis]
                    assert evidence['quote'] and evidence['quote'] in packets[rid]['response']
                    assert evidence['reason'].strip()
        ratings.append(index)
    summary=read(directory/'review-summary.json')
    assert summary['n']==800 and summary['primary_review_n']==384 and summary['secondary_review_n']==416
    for axis in AXES:
        actual=sum(ratings[0][rid][axis]==ratings[1][rid][axis] for rid in key)/800
        assert actual==summary['agreement'][axis]['exact_agreement']
    if (directory/'sha256.json').exists():
        for name,digest in read(directory/'sha256.json').items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    print('Verified 96 held-out prompts, 2016 outputs, 96 restorations, and two complete 800-item blinded model reviews.')

if __name__=='__main__':main()
