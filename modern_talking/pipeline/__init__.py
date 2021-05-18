from csv import DictReader
from itertools import groupby
from json import load, dump
from pathlib import Path
from typing import Set

from modern_talking.evaluation import Metric
from modern_talking.matchers import Matcher
from modern_talking.model import Argument, KeyPoint, Labels, LabelledDataset, \
    DatasetType


data_dir = Path(__file__).parent.parent.parent / "data"
output_dir = data_dir / "out"

class Pipeline:
    """
    Pipeline for training, labelling and evaluation.
    With pipeline instances, we should be able to try out different matchers
    and evaluation metrics easily.
    This class is also be used to parse the dataset files and format the labels
    returned by the matcher.
    """

    matcher: Matcher
    metric: Metric

    def __init__(self, matcher: Matcher, evaluator: Metric):
        self.matcher = matcher
        self.metric = evaluator

    @staticmethod
    def load_dataset(dataset_type: DatasetType) -> LabelledDataset:
        """
        Load a single dataset with arguments, key points and match labels
        from the data directory.
        :param dataset_type: The dataset type to load.
        :return: Parsed, labelled dataset.
        """
        suffix: str
        if dataset_type == DatasetType.TRAIN:
            suffix = "train"
        elif dataset_type == DatasetType.TEST:
            suffix = "test"
        elif dataset_type == DatasetType.DEV:
            suffix = "dev"
        else:
            raise Exception("Unknown dataset type")

        arguments_file = data_dir / f"arguments_{suffix}.csv"
        key_points_file = data_dir / f"key_points_{suffix}.csv"
        labels_file = data_dir / f"labels_{suffix}.csv"

        arguments = Pipeline.load_arguments(arguments_file)
        key_points = Pipeline.load_key_points(key_points_file)
        labels = Pipeline.load_labels(labels_file)

        return LabelledDataset(arguments, key_points, labels)

    @staticmethod
    def load_arguments(path: Path) -> Set[Argument]:
        """
        Load arguments from a CSV file.
        :param path: Path to the CSV file.
        :return: A set of arguments from the file.
        """
        with path.open("r") as file:
            csv = DictReader(file)
            return {
                Argument(row["arg_id"], row["argument"], row["topic"],
                         int(row["stance"]))
                for row in csv
            }

    @staticmethod
    def load_key_points(path: Path) -> Set[KeyPoint]:
        """
        Load key points from a CSV file.
        :param path: Path to the CSV file.
        :return: A set of key points from the file.
        """
        with path.open("r") as file:
            csv = DictReader(file)
            return {
                KeyPoint(row["key_point_id"], row["key_point"], row["topic"],
                         int(row["stance"]))
                for row in csv
            }

    @staticmethod
    def load_labels(path: Path) -> Labels:
        """
        Load argument key point match labels from a CSV file.
        :param path: Path to the CSV file.
        :return: A dictionary of match labels for argument and key point IDs
        from the file.
        """
        with path.open("r") as file:
            csv = DictReader(file)
            return {
                (row["arg_id"], row["key_point_id"]): float(row["label"])
                for row in csv
            }

    @staticmethod
    def load_predictions(path: Path) -> Labels:
        """
        Load predicted argument key point match labels from a JSON file.
        :param path: Path to the JSON file.
        :return: A dictionary of match labels for argument and key point IDs
        from the file.
        """
        with path.open("r") as file:
            json = load(file)
            return {
                (arg, kp): float(label)
                for arg, kps in json.items()
                for kp, label in kps.items()
            }

    @staticmethod
    def save_predictions(path: Path, labels: Labels):
        """
        Save predicted argument key point match labels to a JSON file.
        :param path: Path to the JSON file.
        :param labels: A dictionary of match labels for argument and
        key point IDs to save to the file.
        """
        with path.open("w") as file:
            groups = groupby(labels.keys(), lambda arg_kp: arg_kp[0])
            json = {
                arg: {
                    kp: labels[arg, kp]
                    for _, kp in arg_kp
                }
                for (arg, arg_kp) in groups
            }
            dump(json, file)

    def train_evaluate(self, ignore_test: bool = False) -> float:
        """
        Parse training, test, and development data, train the matcher,
        and evaluate label quality.
        :param ignore_test: If true, use the development dataset
        instead of the test dataset for evaluation.
        This is useful for example when the test dataset is not available
        during model development, like in the shared task.
        :return: The evaluated score as returned by the evaluator.
        """

        # Prepare matcher.
        print("Prepare matcher.")
        self.matcher.prepare()

        # Load datasets.
        print("Load datasets.")
        train_data = Pipeline.load_dataset(DatasetType.TRAIN)
        dev_data = Pipeline.load_dataset(DatasetType.DEV)
        test_data = Pipeline.load_dataset(DatasetType.TEST) \
            if not ignore_test else dev_data

        # Train model.
        print("Train model.")
        self.matcher.train(train_data, dev_data)
        # Predict labels for test data.
        print("Predict labels.")
        predicted_labels = self.matcher.predict(test_data)

        print("Save predictions.")
        predictions_file = output_dir / f"predictions-{self.metric.name}-" \
                                        f"with-{self.matcher.name}.json"
        Pipeline.save_predictions(predictions_file, predicted_labels)
        saved_predicted_labels = Pipeline.load_predictions(predictions_file)
        assert predicted_labels == saved_predicted_labels

        # Get ground-truth labels from test data.
        ground_truth_labels = test_data.labels
        # Evaluate labels.
        print("Evaluate labels.")
        return self.metric.evaluate(predicted_labels, ground_truth_labels)
