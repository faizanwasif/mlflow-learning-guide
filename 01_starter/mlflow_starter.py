from __future__ import annotations

import random
from dataclasses import dataclass

import mlflow


EVAL_EXAMPLES = [(i, i % 2 == 0) for i in range(1, 41)]  # (input, expected_result)


@dataclass
class ModelConfig:
    name: str
    noise: float  # 0.0 = perfect, 1.0 = pure random -- stands in for "model quality"
    bias: float  # simulates a systematic skew, stands in for a badly chosen threshold


def model_score(value: int, config: ModelConfig, rng: random.Random) -> bool:
    
    correct_answer = value % 2 == 0

    if rng.random() < config.noise:
        prediction = not correct_answer 
    else:
        prediction = correct_answer

    if rng.random() < abs(config.bias):
        prediction = config.bias > 0  # biased threshold overrides the call

    return prediction


def evaluate(config: ModelConfig, seed: int = 42) -> dict:
    rng = random.Random(seed)
    correct = 0
    for value, expected in EVAL_EXAMPLES:
        predicted = model_score(value, config, rng)
        if predicted == expected:
            correct += 1
    accuracy = correct / len(EVAL_EXAMPLES)
    return {"accuracy": accuracy, "correct": correct, "total": len(EVAL_EXAMPLES)}


CONFIGS_TO_COMPARE = [
    ModelConfig(name="baseline", noise=0.1, bias=0.0),
    ModelConfig(name="noisier-model", noise=0.4, bias=0.0),
    ModelConfig(name="biased-threshold", noise=0.1, bias=-0.3),
    ModelConfig(name="best-candidate", noise=0.05, bias=0.0),
]


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlflow-starter-demo")

    print(f"Evaluating {len(CONFIGS_TO_COMPARE)} configurations against "
          f"{len(EVAL_EXAMPLES)} labeled examples...\n")

    for config in CONFIGS_TO_COMPARE:
        with mlflow.start_run(run_name=config.name):
            # Parameters: the configuration used for this run.
            mlflow.log_params({
                "config_name": config.name,
                "noise": config.noise,
                "bias": config.bias,
                "eval_set_size": len(EVAL_EXAMPLES),
            })

            result = evaluate(config)

            # Metrics: the numeric results.
            mlflow.log_metrics({
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            })

            # Tags: free-form labels, good for pass/fail-style categorization.
            outcome = (
                "GOOD" if result["accuracy"] >= 0.9
                else "FAIR" if result["accuracy"] >= 0.7
                else "POOR"
            )
            mlflow.set_tag("outcome", outcome)

            print(f"  {config.name:20s} accuracy={result['accuracy']:.2%}  "
                  f"({result['correct']}/{result['total']})  [{outcome}]")

    
    print("\n--- Comparison report (mlflow.search_runs) ---\n")
    runs = mlflow.search_runs(
        experiment_names=["mlflow-starter-demo"],
        order_by=["metrics.accuracy DESC"],
    )
    report_columns = ["params.config_name", "metrics.accuracy", "tags.outcome"]
    print(runs[report_columns].to_string(index=False))

    print("\nDone. Open the MLflow UI to compare these runs visually:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("  -> http://127.0.0.1:5000 -> toggle sidebar to 'Model training' "
          "-> mlflow-starter-demo -> select runs -> Compare")


if __name__ == "__main__":
    main()
