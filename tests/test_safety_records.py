import hashlib
import pytest
from experiments.safety_records import evaluation_rows


def fixture():
    rows = [dict(id='a', label='harmful', text='h', source_tag='sr', row_index=0),
            dict(id='b', label='safe', text='s', source_tag='xs', row_index=0),
            dict(id='c', label='safe', text='fit only', source_tag='xs', row_index=1)]
    split = dict(by_id={'a': 'eval_malicious', 'b': 'eval_benign', 'c': 'fit_benign'},
                 counts={'eval_malicious': 1, 'eval_benign': 1})
    return rows, split


def test_read_frozen_partition_without_fit_leak():
    rows, split = fixture(); result = evaluation_rows(rows, split)
    assert [r['id'] for r in result] == ['a', 'b']
    assert result[0]['prompt_sha256'] == hashlib.sha256(b'h').hexdigest()
    assert split['by_id']['c'] == 'fit_benign'


def test_missing_ids_and_wrong_labels_raise():
    rows, split = fixture()
    with pytest.raises(ValueError): evaluation_rows(rows[:-1], split)
    rows[0]['label'] = 'safe'
    with pytest.raises(ValueError): evaluation_rows(rows, split)


def test_smoke_size_is_bounded():
    rows, split = fixture()
    assert len(evaluation_rows(rows, split, 1)) == 2
    with pytest.raises(ValueError): evaluation_rows(rows, split, 0)
    with pytest.raises(ValueError): evaluation_rows(rows, split, 2)
