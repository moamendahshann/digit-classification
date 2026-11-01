# Handwritten Digit Classification

A compact computer-vision experiment: recognise 8×8 handwritten digits with a scaled SVM pipeline, then inspect both the scores and the mistakes.

[![CI](https://github.com/moamendahshann/digit-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/moamendahshann/digit-classification/actions/workflows/ci.yml)

![Confusion matrix on the held-out test set](results/confusion_matrix.png)

## Results

Recorded run on the bundled scikit-learn digits dataset:

| Measure | Result |
| --- | ---: |
| Held-out accuracy | **98.06%** |
| Macro F1 | 0.9805 |
| Five-fold training CV accuracy | 97.98% ± 0.64 percentage points |
| Majority-class baseline accuracy | 10.00% |
| Train / test samples | 1,437 / 360 |

[Full recorded metrics](results/metrics.json) · [Prediction errors](results/errors.png)

## Method

1. Make a stratified 80/20 split with seed 42.
2. Fit scaling within the SVM pipeline so each cross-validation fold learns its own preprocessing.
3. Run five-fold cross-validation on training data only.
4. Fit the final model with a fixed C=5 and evaluate once on the held-out test set.
5. Compare against a majority-class baseline and save an error gallery.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

The dataset is bundled with scikit-learn; no dataset download or API key is required. Results and plots are written to `results/`. Random seed: 42. Dependency and platform differences can affect results.

## Interpretation and limitations

The dataset is a small classical benchmark. Random stratification does not establish writer-independent performance, and the model is not an OCR service for arbitrary photographs. C is fixed rather than tuned on the test set.

## Development notes

This is an educational portfolio project developed with AI assistance. The recorded results describe the included experiment, not a production system or a previous employment achievement.

## Automated checks

GitHub Actions reruns the experiment and checks the generated metrics and plots. You can run the same check locally with `python scripts/verify_results.py` after training. The status badge reflects the latest workflow result; it is not a real-world quality guarantee.

## Source

[scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
