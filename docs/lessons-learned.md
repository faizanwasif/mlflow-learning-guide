# Lessons Learned: The Metric/Threshold Trap

This is the single most useful mistake made while building this repo, written
up on its own because it's easy to repeat and expensive to debug if you don't
know to look for it.

## The setup

I was comparing several embedding models for an image-similarity task (given
two photos, are they the same subject?). Each model gets converted into a
feature vector, and "same subject" is decided by thresholding a distance
between two vectors.

The evaluation compared multiple models side by side, all scored the same
way, to see which performed best.

## The mistake

I picked one distance metric (cosine distance) and applied it uniformly to
every model being compared, using what looked like a single reasonable
threshold. This seemed fair — same yardstick for everyone.

One model came back dramatically worse than the others. Tens of percentage
points worse. It looked like a real, actionable finding: "this model is bad,
don't use it."

## Why it was wrong

Different embedding models (and different distance metrics on the *same*
model) don't share a common scale. Cosine distance is bounded to roughly
`[0, 2]`; raw Euclidean distance on unnormalized vectors can be anywhere,
often much larger. A threshold that's sensibly calibrated for one
(model, metric) pair can be nonsensically strict or loose for another.

The "obviously worse" model wasn't worse — it was tuned for a different
metric than the one I forced on it. Under its own natural metric, it
performed in line with everything else.

## How I caught it

I had an independent reference: a previous, separately-run test using the
*correct* production configuration for that one model. The numbers didn't
match at all — not "a bit different," but a large enough discrepancy that
something had to be wrong. That mismatch is what triggered the investigation
that found the bug.

**The general lesson:** an unexpectedly large gap between configurations is
itself a signal to double-check the evaluation, not just the models. If you
don't have an independent reference to compare against, the fallback is to
sweep every reasonable metric per model (cheap, since only the distance
computation is re-run — not the expensive embedding step) rather than pick
one metric and assume it's fair to everyone.

## The fix, structurally

`evaluate_similarity_models.py` in this repo bakes the fix in from the start:

- Every `(embedder, metric)` combination gets its **own** threshold, in an
  explicit lookup table (`THRESHOLDS`), rather than one threshold applied
  everywhere.
- Every combination is logged as its own MLflow run, so nothing is silently
  averaged together or compared unfairly.
- Embeddings are computed once per image and reused across every metric
  (the expensive step doesn't need to be repeated to be fair).

Run it and look at how much accuracy shifts for the *same* embedder just by
changing the metric — that variation is the whole point of this example.

## The takeaway for anyone using MLflow to compare models

Before trusting a run comparison in the MLflow UI:

1. Check that every run's configuration is genuinely comparable — not just
   "same code path," but same units, same calibration, same assumptions.
2. If one run is a dramatic outlier, treat that as a prompt to re-check the
   evaluation itself before writing it up as a finding about the model.
3. When a library or model surfaces multiple metrics/modes, assume they are
   **not** interchangeable unless you've confirmed otherwise — check the
   library's own documented defaults per mode rather than picking one and
   applying it everywhere.
