# Getting Started with MLflow: A Practical Guide

This guide walks through setting up MLflow and tracking your first real model
experiments. It's written from an actual first run-through, including the
mistakes that came up along the way — so the "gotchas" sections aren't
theoretical, they're things that will probably happen to you too.

Two runnable examples live alongside this guide:
- [`01_starter/mlflow_starter.py`](../01_starter/mlflow_starter.py) — a
  zero-dependency (besides MLflow itself) toy sweep. Start here.
- [`02_image_similarity_eval/`](../02_image_similarity_eval/) — a more
  realistic multi-model x multi-metric evaluation harness, plus a synthetic
  dataset generator so it runs with no real data. See
  [lessons-learned.md](lessons-learned.md) for the mistake that shaped its design.

---

## 1. What MLflow actually does

MLflow is a system for tracking machine learning experiments. The core idea:
every time you run a model with some configuration (a model choice, a
threshold, a hyperparameter), you log it as a **run** — its inputs
(parameters), its results (metrics), and any output files (artifacts). Once
you have several runs, you can compare them side by side instead of relying
on memory or scattered print statements.

The part that matters most for a beginner: **one run in isolation isn't
useful.** MLflow's value only shows up once you have multiple runs to
compare. If you only ever run your script once, you'll wonder what the point
was — so the goal from day one should be "run it several times with
different settings," not "run it once and look at the dashboard."

---

## 2. Install MLflow

```bash
pip install mlflow
# or, if you use uv:
uv pip install mlflow
```

Use a dedicated virtual environment (venv or conda) for whatever project
you're instrumenting — don't install into a shared/base environment.

---

## 3. Set a tracking backend (don't skip this)

MLflow needs somewhere to store run data.

Use a SQLite-backed URI — it's one line, requires no server, and is
the currently recommended default:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

---

## 4. Write your first tracked run

This is the actual shape of a minimal script:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my-first-experiment")  # groups related runs, like a folder

with mlflow.start_run(run_name="baseline"):
    # Parameters: the configuration you used
    mlflow.log_params({
        "model_name": "my-model-v1",
        "threshold": 0.5,
    })

    # ... run your actual model/logic here ...
    accuracy = 0.87  # whatever your real result is

    # Metrics: the numeric results
    mlflow.log_metrics({
        "accuracy": accuracy,
    })

    # Tags: free-form labels, good for pass/fail or categorical outcomes
    mlflow.set_tag("outcome", "PASSED" if accuracy > 0.8 else "FAILED")
```

Run it:

```bash
python my_script.py
```

---

## 5. Open the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open **http://127.0.0.1:5000** in your browser.

**Gotcha — the UI has two modes.** Recent MLflow UI versions default to a
"GenAI" view (built for LLM tracing — traces, sessions, judges) rather than
the classic experiment tracking view. Look for a small toggle near the top
of the left sidebar labeled **GenAI / Model training**. Click **Model
training** to get the classic Experiments → Runs → Parameters/Metrics view
described in this guide. If you only see "0 traces, no data available," you're
almost certainly in the wrong mode, not looking at a broken install.

---

## 6. Make it actually useful: run it more than once

A single run just shows you one number. The real workflow is a **sweep**:
run the same logic multiple times with one thing changed, so you can compare.

```python
for model_name in ["model-a", "model-b", "model-c"]:
    with mlflow.start_run(run_name=f"eval-{model_name}"):
        mlflow.log_params({"model_name": model_name})
        # ... run and evaluate ...
        mlflow.log_metrics({"accuracy": accuracy})
```

In the UI: go to your experiment, select the checkboxes next to multiple
runs, and click **Compare**. This is where MLflow earns its keep — you get a
table (and charts) of every run's parameters and metrics side by side.

---

## 7. Build a real evaluation, not just a demo

The most useful beginner exercise isn't "log a number," it's: **evaluate
something against a known ground truth so you get a real accuracy metric.**

If you're testing a classifier, matcher, or similarity model, structure it like
this:
1. Get a labeled dataset — a set of inputs where you already know the correct
   answer.
2. Run your model against every labeled example.
3. Compare the model's prediction to the known answer.
4. Log `accuracy = correct / total` as a metric, not just raw scores.

This turns MLflow from "a place where numbers get stored" into "a tool that
tells you which configuration is actually better" — which is the whole point.

---

## 8. The trap that will get you: mismatched thresholds/metrics

This is a real mistake made while building this guide, worth calling out
explicitly: if your model/library supports multiple distance metrics or
scoring modes (e.g. cosine vs. euclidean distance), **each one usually has
its own separate, calibrated threshold.** Using one metric's threshold with
a different metric's distance values will silently produce nonsense
results — and it will look like a legitimate finding ("Model X is much worse
than Model Y!") when it's actually just a bug in the evaluation script.

Symptoms this happened to you:
- One "model" or "configuration" scores dramatically worse than all the
  others, with no obvious reason why.
- The gap is too large to be believable (e.g. 40+ percentage points).

Before trusting a surprising result: check whether every configuration is
being scored with the *matching* threshold/metric it was actually tuned for,
not one forced uniformly across everything. When in doubt, sweep every
combination (all models × all metrics) rather than picking one and assuming
it's fair to all of them — that's usually cheap to do, since the expensive
part (running the model) doesn't need to be repeated, only the
scoring/comparison step.

---

## 9. Logging a dataset (optional, but more rigorous)

If your evaluation set is meaningful enough to reuse, log it as a proper
MLflow Dataset rather than just implying it in code:

```python
from mlflow.data.pandas_dataset import from_pandas
import pandas as pd

manifest = pd.DataFrame({"example_id": [...], "notes": [...]})
dataset = from_pandas(manifest, name="my-eval-set")

with mlflow.start_run():
    mlflow.log_input(dataset, context="evaluation")
    ...
```

This shows up under the run's "Datasets" section in the UI instead of
`None`, and documents exactly what data produced a given result.

---

## 10. Querying runs programmatically (for reports)

The UI is fine for browsing, but for anything you want to export, share, or
turn into a written report, pull runs into a DataFrame:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(
    experiment_names=["my-first-experiment"],
    order_by=["start_time DESC"],
)
print(runs[["run_id", "status", "params.model_name", "metrics.accuracy"]])
```

From there, export to CSV, build a table, or write a summary — MLflow itself
doesn't generate reports for you, but `search_runs()` is the bridge between
"data MLflow tracked" and "something you can actually share."

---

## 11. Common commands, all in one place

```bash
# Install
pip install mlflow

# Run a script that logs to sqlite
python my_script.py

# Launch the UI against the same backend the script used
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# If using conda envs, run everything inside the env explicitly:
conda run -n my-env python my_script.py
conda run -n my-env mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Summary checklist for your first real experiment

- [ ] Install MLflow into a project-specific environment
- [ ] Use `sqlite:///mlflow.db`, not the plain filesystem backend
- [ ] Log params (config) and metrics (results) for every run
- [ ] Run it more than once with something changed each time
- [ ] Open the UI and switch to **Model training** mode if you see traces/sessions instead of runs
- [ ] Select multiple runs and use **Compare** before trusting any single result
- [ ] If results look suspiciously bad or good for one configuration, double-check that every configuration was scored fairly (same/matching thresholds, same ground truth, same preprocessing)
- [ ] Use `mlflow.search_runs()` to pull results into a table for sharing
