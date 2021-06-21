from itertools import islice
from typing import List, Tuple

from modern_talking.evaluation import Metric
from modern_talking.model import Labels, ArgumentKeyPointIdPair, Label


class ManualErrors(Metric):
    name = "manual-errors"

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> float:
        ids = Metric.get_all_ids(predicted_labels, ground_truth_labels)
        samples: List[Tuple[ArgumentKeyPointIdPair, Label, Label, float]] = []
        for arg, kp in ids:
            true_label = ground_truth_labels[arg, kp]
            pred_label = predicted_labels[arg, kp]
            error = abs(true_label - pred_label)
            samples.append(((arg, kp), true_label, pred_label, error))
        samples = list(sorted(
            samples,
            key=lambda sample: sample[3],
            reverse=True,
        ))
        if len(samples) > 50:
            print("Showing only worst 50 pairs.")
        for (arg_kp, true_label, pred_label, error) in islice(samples, 50):
            arg, kp = arg_kp
            print(
                f"Error {error} for {arg} and {kp} "
                f"(predicted: {pred_label}, true: {true_label})")
        return -1
