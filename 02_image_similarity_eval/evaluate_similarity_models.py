from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
from PIL import Image

IMAGE_ROOT = Path(__file__).resolve().parent / "synthetic_dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MIN_IMAGES_PER_IDENTITY = 3
IMAGES_PER_IDENTITY = 3


# ---------------------------------------------------------------------------
# "Models" -- each is just a different way of turning an image into a feature
# vector. Swap these for real embedding models (DeepFace, CLIP, etc.) without
# changing anything below this section.
# ---------------------------------------------------------------------------

def embed_mean_color(img: Image.Image) -> np.ndarray:
    """Feature = average R,G,B. Cheap, ignores shape/position entirely."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return arr.reshape(-1, 3).mean(axis=0)


def embed_color_histogram(img: Image.Image) -> np.ndarray:
    """Feature = coarse per-channel color histogram. Captures more than the mean."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    bins_per_channel = 8
    hist = np.concatenate([
        np.histogram(arr[:, :, c], bins=bins_per_channel, range=(0, 255))[0]
        for c in range(3)
    ]).astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def embed_downsampled_pixels(img: Image.Image) -> np.ndarray:
    """Feature = flattened low-res thumbnail. Captures rough shape/position too."""
    small = img.convert("L").resize((16, 16))
    return np.asarray(small, dtype=np.float32).flatten() / 255.0


EMBEDDERS: dict[str, Callable[[Image.Image], np.ndarray]] = {
    "mean-color": embed_mean_color,
    "color-histogram": embed_color_histogram,
    "downsampled-pixels": embed_downsampled_pixels,
}


# ---------------------------------------------------------------------------
# Distance metrics -- each has its OWN natural scale, which is exactly the
# point of this example. A "same person" threshold tuned for cosine distance
# (roughly 0-2) means nothing applied to euclidean distance on raw pixel
# vectors (which can be in the hundreds).
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(1 - np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


DISTANCE_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
}

# A threshold appropriate for one (embedder, metric) pair is usually wrong
# for another -- this table makes that explicit instead of hiding it.
# In a real project you'd calibrate these against a held-out labeled set;
# here they're picked to roughly work so the demo has something to show.
THRESHOLDS: dict[tuple[str, str], float] = {
    ("mean-color", "cosine"): 0.02,
    ("mean-color", "euclidean"): 20.0,
    ("color-histogram", "cosine"): 0.15,
    ("color-histogram", "euclidean"): 0.25,
    ("downsampled-pixels", "cosine"): 0.05,
    ("downsampled-pixels", "euclidean"): 3.0,
}


@dataclass
class PairResult:
    identity_id: str
    pair: str
    embedder: str
    metric: str
    distance: float
    threshold: float
    predicted_same_identity: bool
    expected_same_identity: bool = True  # every pair here is same-identity by construction


def discover_identity_folders(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        d for d in sorted(root.iterdir())
        if d.is_dir()
        and len([f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
        >= MIN_IMAGES_PER_IDENTITY
    ]


def evaluate(identity_folders: list[Path]) -> list[PairResult]:
    results = []
    for folder in identity_folders:
        images = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )[:IMAGES_PER_IDENTITY]
        loaded = [Image.open(p) for p in images]

        # Embed once per (image, embedder) -- reused across every metric,
        # since embedding is the expensive part and distance computation is not.
        embeddings = {
            embedder_name: [embedder_fn(img) for img in loaded]
            for embedder_name, embedder_fn in EMBEDDERS.items()
        }

        for embedder_name, embs in embeddings.items():
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    for metric_name, metric_fn in DISTANCE_METRICS.items():
                        distance = metric_fn(embs[i], embs[j])
                        threshold = THRESHOLDS[(embedder_name, metric_name)]
                        results.append(PairResult(
                            identity_id=folder.name,
                            pair=f"img{i}_vs_img{j}",
                            embedder=embedder_name,
                            metric=metric_name,
                            distance=distance,
                            threshold=threshold,
                            predicted_same_identity=distance <= threshold,
                        ))
    return results


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("image-similarity-eval")

    identity_folders = discover_identity_folders(IMAGE_ROOT)
    if not identity_folders:
        print(f"No identity folders found under {IMAGE_ROOT}.")
        print("Run generate_synthetic_dataset.py first, or point IMAGE_ROOT "
              "at your own 'one folder per identity' dataset.")
        return

    print(f"Found {len(identity_folders)} identities. "
          f"Evaluating {len(EMBEDDERS)} embedders x {len(DISTANCE_METRICS)} metrics...\n")

    all_results = evaluate(identity_folders)

    for embedder_name in EMBEDDERS:
        for metric_name in DISTANCE_METRICS:
            subset = [
                r for r in all_results
                if r.embedder == embedder_name and r.metric == metric_name
            ]
            correct = sum(r.predicted_same_identity == r.expected_same_identity for r in subset)
            accuracy = correct / len(subset) if subset else 0.0
            avg_distance = sum(r.distance for r in subset) / len(subset) if subset else 0.0

            with mlflow.start_run(run_name=f"{embedder_name}-{metric_name}"):
                mlflow.log_params({
                    "embedder": embedder_name,
                    "metric": metric_name,
                    "threshold": THRESHOLDS[(embedder_name, metric_name)],
                    "num_identities": len(identity_folders),
                    "images_per_identity": IMAGES_PER_IDENTITY,
                })
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "avg_distance": avg_distance,
                    "num_pairs": float(len(subset)),
                })
                outcome = "GOOD" if accuracy >= 0.9 else "FAIR" if accuracy >= 0.7 else "POOR"
                mlflow.set_tag("outcome", outcome)

                print(f"  {embedder_name:20s} {metric_name:10s} "
                      f"accuracy={accuracy:.2%}  avg_distance={avg_distance:.4f}  [{outcome}]")

    print("\n--- Comparison report ---\n")
    runs = mlflow.search_runs(
        experiment_names=["image-similarity-eval"],
        order_by=["metrics.accuracy DESC"],
    )
    print(runs[["params.embedder", "params.metric", "metrics.accuracy", "tags.outcome"]]
          .to_string(index=False))

    print("\nNotice how much accuracy varies by (embedder, metric) pair on the SAME")
    print("images -- see docs/lessons-learned.md for why that's the whole point.")
    print("\nOpen the UI to compare visually:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")


if __name__ == "__main__":
    main()
