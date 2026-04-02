"""Example pipelines demonstrating usage of models_selection."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_models_selection import models_selection


@dsl.pipeline(name="autogluon-models-selection-example")
def example_pipeline(
    label_column: str = "target",
    task_type: str = "regression",
    top_n: int = 3,
    train_data_path: str = "/tmp/train_data",
    workspace_path: str = "/tmp/workspace",
):
    """Example pipeline using models_selection.

    Args:
        label_column: Name of the label column.
        task_type: Type of ML task.
        top_n: Number of top models to select.
        train_data_path: Path to the training data.
        workspace_path: Path to the workspace directory.
    """
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Dataset,
    )
    models_selection(
        label_column=label_column,
        task_type=task_type,
        top_n=top_n,
        train_data_path=train_data_path,
        test_data=test_data.output,
        workspace_path=workspace_path,
    )
