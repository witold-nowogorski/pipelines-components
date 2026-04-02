"""Example pipelines demonstrating usage of autogluon_models_full_refit."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_models_full_refit import (
    autogluon_models_full_refit,
)


@dsl.pipeline(name="autogluon-models-full-refit-example")
def example_pipeline(
    model_name: str = "WeightedEnsemble_L2",
    predictor_path: str = "/tmp/predictor",
    pipeline_name: str = "automl-pipeline",
    run_id: str = "run-001",
    sample_row: str = '{"feature1": 1.0}',
    extra_train_data_path: str = "",
):
    """Example pipeline using autogluon_models_full_refit.

    Args:
        model_name: Name of the model to refit.
        predictor_path: Path to the predictor.
        pipeline_name: Name of the pipeline.
        run_id: Unique run identifier.
        sample_row: JSON string of a sample row.
        extra_train_data_path: Path to extra training data.
    """
    test_dataset = dsl.importer(
        artifact_uri="gs://placeholder/test_dataset",
        artifact_class=dsl.Dataset,
    )
    autogluon_models_full_refit(
        model_name=model_name,
        test_dataset=test_dataset.output,
        predictor_path=predictor_path,
        pipeline_name=pipeline_name,
        run_id=run_id,
        sample_row=sample_row,
        extra_train_data_path=extra_train_data_path,
    )
