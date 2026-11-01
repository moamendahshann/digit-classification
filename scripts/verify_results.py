"""Smoke-check the generated experiment artifacts, not real-world model quality."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results'
m = json.loads((root / 'metrics.json').read_text())
assert m['train_samples'] + m['test_samples'] == 1797
assert .90 < m['test_accuracy'] <= 1
assert m['baseline_accuracy'] < m['test_accuracy']
assert 0 <= m['macro_f1'] <= 1
plots = ['confusion_matrix.png', 'errors.png']
for name in plots:
    data = (root / name).read_bytes()
    assert data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) > 100
print('Experiment metrics and plots passed sanity checks.')
