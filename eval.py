from modern_talking.data import download_kpa_2021_data
from modern_talking.matchers.utils import setup_colab_tpu
from modern_talking.pipeline import Pipeline
from modern_talking.pipeline.cli import parse_pipeline_cli


def evaluate(pipeline: Pipeline) -> None:
    """
    Evaluate matcher.
    """
    print(f"Train/evaluate matcher '{pipeline.matcher.slug}' "
          f"with metric '{pipeline.metric.slug}'.")

    setup_colab_tpu()

    # Download datasets.
    download_kpa_2021_data()

    # Execute pipeline.
    result = pipeline.evaluate()

    print(f"Final score for metric {pipeline.metric.slug}: {result:.3f}")


if __name__ == "__main__":
    evaluate(parse_pipeline_cli())
