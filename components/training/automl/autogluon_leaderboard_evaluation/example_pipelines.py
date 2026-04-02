"""Example pipelines demonstrating usage of leaderboard_evaluation."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import (
    leaderboard_evaluation,
)


@dsl.component
def produce_model(name: str, model: dsl.Output[dsl.Model]):
    """Produce a dummy model artifact for demonstration purposes."""
    import json
    import os

    metrics_dir = os.path.join(model.path, name, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump({"root_mean_squared_error": 0.5}, f)
    model.metadata["display_name"] = name


@dsl.pipeline(name="autogluon-leaderboard-evaluation-example")
def example_pipeline(
    eval_metric: str = "root_mean_squared_error",
):
    """Example pipeline using leaderboard_evaluation.

    Args:
        eval_metric: Evaluation metric name.
    """
    with dsl.ParallelFor(items=["ModelA", "ModelB"]) as model_name:
        model_task = produce_model(name=model_name)

    leaderboard_evaluation(
        models=dsl.Collected(model_task.outputs["model"]),
        eval_metric=eval_metric,
    )
