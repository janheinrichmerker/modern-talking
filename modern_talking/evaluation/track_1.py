from pathlib import Path
from tempfile import TemporaryDirectory

from modern_talking.evaluation import Metric
from modern_talking.evaluation.track_1_kp_matching import load_kpm_data, \
    get_predictions, calc_mean_average_precision
from modern_talking.model import Labels
from modern_talking.pipeline import Pipeline


class Track1Metric(Metric):

    def __init__(self, relaxed: bool = False):
        self.relaxed = relaxed

    @property
    def name(self) -> str:
        variant = "relaxed" if self.relaxed else "strict"
        return f"track-1-{variant}"

    def evaluate(self, predicted_labels: Labels,
                 ground_truth_labels: Labels) -> float:
        gold_data_dir = Path(__file__).parent.parent.parent / "data"
        arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset="dev")

        with TemporaryDirectory() as temp_dir:
            predictions_file = Path(temp_dir) / "predictions.json"
            Pipeline.save_predictions(predictions_file, predicted_labels)

            merged_df = get_predictions(predictions_file, labels_df, arg_df)
            return calc_mean_average_precision(
                merged_df, "label_relaxed" if self.relaxed else "label_strict")
