from modern_talking.data import download_kpa_2021_data
from modern_talking.matchers.utils import setup_colab_tpu
from modern_talking.pipeline import Pipeline
from modern_talking.pipeline.cli import parse_pipeline_cli


def train_evaluate(pipeline: Pipeline) -> None:
    """
    Train/evaluate matcher.
    """
    print(f"Train/evaluate matcher '{pipeline.matcher.slug}' "
          f"with metric '{pipeline.metric.slug}'.")

    setup_colab_tpu()

    # Download datasets.
    download_kpa_2021_data()

    # Execute pipeline.
    result = pipeline.train_evaluate()

    print(f"Final score for metric {pipeline.metric.slug}: {result:.4f}")


if __name__ == "__main__":
    train_evaluate(parse_pipeline_cli())
