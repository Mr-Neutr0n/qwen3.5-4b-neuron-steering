import json
from pathlib import Path
import pytest
from study_analysis import checks, strict_json_equal, bootstrap_family, reviewed_summary

def test_content_and_format_are_separate():
    p={'category':'capability','check':'exact','expected':'254'}
    assert checks(p,'254')['strict_correct']
    assert checks(p,'Answer: 254\nextra text')==dict(strict_correct=False,first_answer_correct=True,format_compliant=False)
    assert checks(p,'17')==dict(strict_correct=False,first_answer_correct=False,format_compliant=True)
    assert not checks(p,'254 is wrong; use 17')['first_answer_correct']

def test_json_rejects_bool_numeric_and_extra_content():
    p={'category':'capability','check':'json','expected_object':{'count':1}}
    assert not strict_json_equal({'count':True},{'count':1})
    r=checks(p,'{"count":1}\nHere you go')
    assert r==dict(strict_correct=False,first_answer_correct=True,format_compliant=False)
    assert checks(p,'{"count":2}')['format_compliant']
    assert not checks(p,'{"count":2}')['strict_correct']

def test_bootstrap_clusters_paraphrases_as_one_unit():
    a=bootstrap_family({'one':[0,0,0,0],'two':[1,1,1,1]})
    b=bootstrap_family({'one':[0],'two':[1]})
    assert a==b and a['families']==2 and a['mean']==.5

def test_reviews_require_complete_blind_ids(tmp_path):
    (tmp_path/'blind-key.json').write_text(json.dumps({'a':{'prompt_id':'p','condition':'baseline','category':'harmful','family':'f'}}))
    for name in ['a.json','b.json']:(tmp_path/name).write_text('[]')
    with pytest.raises(ValueError,match='every blinded record'):
        reviewed_summary(tmp_path,tmp_path/'a.json',tmp_path/'b.json')
