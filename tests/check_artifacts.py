"""Check model/header identity and saved references without TensorFlow."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

root = Path(__file__).resolve().parents[1]
artifacts = root / 'training/artifacts'
report = json.loads((artifacts / 'conversion_report.json').read_text())
for kind in ('fp32', 'int8'):
    data = (artifacts / f'cnn_{kind}.tflite').read_bytes()
    assert len(data) == report[f'{kind}_file_bytes']
    assert hashlib.sha256(data).hexdigest() == report[f'{kind}_sha256']

header = (root / 'firmware/main/cnn_model_data.h').read_text()
embedded = bytes(int(x, 16) for x in re.findall(r'0x([0-9a-fA-F]{2})', header))
assert embedded == (artifacts / 'cnn_int8.tflite').read_bytes()

fixtures = (root / 'firmware/main/cnn_test_data.h').read_text()
def array(name):
    body = re.search(r'\b' + name + r'\[[^;]+?=\s*\{(.*?)\};', fixtures, re.S)
    assert body, name
    return np.array([int(x) for x in re.findall(r'-?\d+', body.group(1))])

labels = array('labels')
indices = array('indices')
scores = array('expected').reshape(100, 10)
pixels = array('images').reshape(100, 784)
assert ((pixels >= 0) & (pixels <= 255)).all()
assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, 10))
with np.load(artifacts / 'tflite_test_reference.npz', allow_pickle=False) as ref:
    assert np.array_equal(labels, ref['labels'][indices])
    assert np.array_equal(scores, ref['int8_outputs'][indices])
    assert int(np.sum(ref['int8_predictions'] == ref['labels'])) == 9858
    assert int(np.sum(ref['fp32_predictions'] == ref['labels'])) == 9861
    assert np.array_equal(np.argmax(ref['int8_outputs'], axis=1), ref['int8_predictions'])
with np.load(artifacts / 'split_indices.npz', allow_pickle=False) as split:
    combined = np.concatenate([split['training'], split['validation']])
    assert np.array_equal(np.sort(combined), np.arange(60000))
    calibration = np.load(artifacts / 'calibration_indices.npy', allow_pickle=False)
    assert len(np.unique(calibration)) == 500
    assert np.isin(calibration, split['training']).all()
print('PASS: model hashes, embedded model, test references, split and calibration indices.')
