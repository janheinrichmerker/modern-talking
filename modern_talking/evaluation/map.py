from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from modern_talking.evaluation import Metric, EvaluationMode
from modern_talking.model import Labels
from modern_talking.pipeline import Pipeline


class MeanAveragePrecision(Metric):
    name = "map"

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels,
            mode: EvaluationMode,
    ) -> float:
        script_dir = Path(__file__).parent
        evaluation_script = script_dir / "track_1_kp_matching.py"
        gold_data_dir = script_dir.parent.parent / "data"
        with TemporaryDirectory() as temp_dir:
            predictions_file = Path(temp_dir) / "predictions.json"
            Pipeline.save_predictions(predictions_file, predicted_labels)

            # Call script file.
            result = run(
                [
                    "python",
                    evaluation_script.absolute(),
                    gold_data_dir.absolute(),
                    predictions_file.absolute(),
                ],
                capture_output=True,
                text=True,
            )

            # Parse result.
            result = str(result.stdout)
            result = result.splitlines()[-1].split(";")
            result_strict = float(result[0].split("=")[1].strip())
            result_relaxed = float(result[1].split("=")[1].strip())

        if mode == EvaluationMode.relaxed:
            return result_relaxed
        else:
            return result_strict
