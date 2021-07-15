from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from modern_talking.evaluation import Metric, EvaluationMode
from modern_talking.evaluation.track_1_kp_matching import \
    calc_mean_average_precision, get_predictions, load_kpm_data
from modern_talking.model import Labels
from modern_talking.pipeline import Pipeline


class MeanAveragePrecision(Metric):
    new: bool

    def __init__(self, new: bool = False):
        self.new = new

    @property
    def slug(self) -> str:
        new_suffix = "-new" if self.new else ""
        return f"map{new_suffix}"

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels,
            mode: EvaluationMode,
    ) -> float:
        if mode == EvaluationMode.relaxed:
            field = "label_relaxed"
        else:
            field = "label_strict"

        labels_count = len(ground_truth_labels)
        if labels_count == 20635:
            subset = "train"
        elif labels_count == 3458:
            subset = "dev"
        elif labels_count == 3426:
            subset = "test"
        else:
            raise Exception(f"Can't detect data subset "
                            f"from label count {labels_count}.")

        gold_data_dir = Path(__file__).parent.parent.parent / "data"

        ignore = StringIO()
        with redirect_stdout(ignore):
            arg_df, kp_df, labels_df = load_kpm_data(
                gold_data_dir,
                subset=subset
            )

        with TemporaryDirectory() as temp_dir:
            predictions_file = Path(temp_dir) / "predictions.json"
            Pipeline.save_predictions(predictions_file, predicted_labels)

            with redirect_stdout(ignore):
                merged_df = get_predictions(
                    predictions_file,
                    labels_df,
                    arg_df,
                    kp_df
                )
            score = calc_mean_average_precision(merged_df, field)
            return score
