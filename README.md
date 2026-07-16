# MLflow Learning Guide

A practical, from-scratch walkthrough of MLflow Tracking — written while
actually learning it, including the mistakes that came up along the way.
Everything here is runnable with synthetic/toy data; no real datasets are
required or included.

## Structure

```
docs/
  beginner-guide.md       Start here — the full walkthrough
  lessons-learned.md       One real mistake worth learning from: mismatched
                            distance metrics/thresholds silently producing
                            a fake "finding"
01_starter/
  mlflow_starter.py        Minimal, dependency-light (just mlflow) example:
                            params, metrics, tags, a 4-way sweep, and a
                            comparison report — using a small synthetic
                            "model" so there's nothing else to install
02_image_similarity_eval/
  generate_synthetic_dataset.py   Generates a small fully-synthetic image
                                   dataset (simple colored shapes, no real
                                   photos) so the eval script below is
                                   runnable by anyone
  evaluate_similarity_models.py   A more realistic multi-model x
                                   multi-metric MLflow evaluation harness,
                                   dataset-agnostic — point it at your own
                                   "one folder per identity" dataset instead
                                   of the synthetic one if you have one
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1. The toy sweep -- no dataset needed
python 01_starter/mlflow_starter.py

# 2. The more realistic evaluation harness
python 02_image_similarity_eval/generate_synthetic_dataset.py
python 02_image_similarity_eval/evaluate_similarity_models.py

# 3. Look at the results
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# open http://127.0.0.1:5000
# NOTE: the sidebar defaults to a "GenAI" view. Toggle it to
# "Model training" to see the classic runs/params/metrics view described
# in the guide.
```
