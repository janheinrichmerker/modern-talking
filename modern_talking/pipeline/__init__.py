from csv import DictReader
from json import load, dump
from math import isnan
from pathlib import Path
from typing import Set
from zipfile import ZipFile

from modern_talking.evaluation import Metric, EvaluationMode
from modern_talking.matchers import Matcher
from modern_talking.model import Argument, KeyPoint, Labels, LabelledDataset, \
    DatasetType, Dataset

data_dir = Path(__file__).parent.parent.parent / "data"
output_dir = data_dir / "out"
cache_dir = data_dir / "cache"


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
    def load_dataset(dataset_type: DatasetType) -> Dataset:
        """
        Load a single dataset with arguments and key points
        from the data directory.
        If the file exists, the match labels are also parsed.
        :param dataset_type: The dataset type to load.
        :return: Parsed (possibly labelled) dataset.
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
        if labels_file.exists():
            labels = Pipeline.load_labels(labels_file)
            return LabelledDataset(arguments, key_points, labels)
        else:
            return Dataset(arguments, key_points)

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
        args = sorted(set(arg for arg, _ in labels.keys()))
        kps = sorted(set(kp for _, kp in labels.keys()))

        with path.open("w") as file:
            json = {
                arg: {
                    kp: labels[arg, kp]
                    for kp in kps
                    if (arg, kp) in labels.keys()
                }
                for arg in args
            }
            dump(json, file)

    @staticmethod
    def save_predictions_archive(predictions_path: Path, zip_path: Path):
        """
        Save a copy of the predictions JSON file in a ZIP file.
        In the archive the file is named `predictions.p`.
        """
        with ZipFile(zip_path, "w") as zip:
            zip.write(predictions_path, "predictions.p")

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
        assert isinstance(train_data, LabelledDataset)
        dev_data = Pipeline.load_dataset(DatasetType.DEV)
        assert isinstance(dev_data, LabelledDataset)
        test_data: Dataset = Pipeline.load_dataset(DatasetType.TEST) \
            if not ignore_test else dev_data

        # Load/train model.
        matcher_path = cache_dir / self.matcher.name
        model_path = matcher_path / "model"
        cache_path = matcher_path / "cache"
        print("Load model.")
        if not self.matcher.load_model(model_path):
            print("Train model.")
            self.matcher.train(train_data, dev_data, cache_path)
            print("Save model.")
            self.matcher.save_model(model_path)

        # Predict labels.
        print("Predict labels.")
        train_labels = self.matcher.predict(train_data)
        dev_labels = self.matcher.predict(dev_data)
        test_labels = self.matcher.predict(test_data)
        print(f"Predicted {len(train_labels)} on train set, "
              f"{len(dev_labels)} on validation set, "
              f"{len(test_labels)} on test set")

        print("Save test predictions.")
        predictions_file = output_dir / f"predictions-{self.matcher.name}.json"
        archive_file = predictions_file.with_suffix(".zip")
        Pipeline.save_predictions(predictions_file, test_labels)
        Pipeline.save_predictions_archive(predictions_file, archive_file)
        saved_test_labels = Pipeline.load_predictions(predictions_file)
        assert saved_test_labels == test_labels

        # Evaluate labels.
        print("Evaluate labels.")
        train_result_strict = self.metric.evaluate(
            train_labels,
            train_data.labels,
            EvaluationMode.strict,
        )
        train_result_relaxed = self.metric.evaluate(
            train_labels,
            train_data.labels,
            EvaluationMode.relaxed,
        )
        train_result_average = (train_result_strict + train_result_relaxed) / 2
        print(
            f"Metric {self.metric.name} on train dataset:"
            f" {train_result_strict:.4f} (strict)"
            f" {train_result_relaxed:.4f} (relaxed)"
            f" {train_result_average:.4f} (average)"
        )

        dev_result_strict = self.metric.evaluate(
            dev_labels,
            dev_data.labels,
            EvaluationMode.strict,
        )
        dev_result_relaxed = self.metric.evaluate(
            dev_labels,
            dev_data.labels,
            EvaluationMode.relaxed,
        )
        dev_result_average = (dev_result_strict + dev_result_relaxed) / 2
        print(
            f"Metric {self.metric.name} on dev dataset:"
            f" {dev_result_strict:.4f} (strict)"
            f" {dev_result_relaxed:.4f} (relaxed)"
            f" {dev_result_average:.4f} (average)"
        )

        if not isinstance(test_data, LabelledDataset):
            print("Metric {self.metric.name} on train dataset: "
                  "skipped because no ground truth labels were found")
            return dev_result_average

        test_result_strict = self.metric.evaluate(
            test_labels,
            test_data.labels,
            EvaluationMode.strict,
        )
        test_result_relaxed = self.metric.evaluate(
            test_labels,
            test_data.labels,
            EvaluationMode.relaxed,
        )
        test_result_average = (test_result_strict + test_result_relaxed) / 2
        saved_test_result_strict = self.metric.evaluate(
            dev_labels,
            dev_data.labels,
            EvaluationMode.strict,
        )
        saved_test_result_relaxed = self.metric.evaluate(
            saved_test_labels,
            test_data.labels,
            EvaluationMode.relaxed,
        )
        saved_test_result_average = (saved_test_result_strict
                                     + saved_test_result_relaxed) / 2
        assert (saved_test_result_strict == test_result_strict
                or (isnan(saved_test_result_strict)
                    and isnan(test_result_strict)))
        assert (saved_test_result_relaxed == test_result_relaxed
                or (isnan(saved_test_result_relaxed)
                    and isnan(test_result_relaxed)))
        assert (saved_test_result_average == test_result_average
                or (isnan(saved_test_result_average)
                    and isnan(test_result_average)))
        print(
            f"Metric {self.metric.name} on test dataset:"
            f" {test_result_strict:.4f} (strict)"
            f" {test_result_relaxed:.4f} (relaxed)"
            f" {test_result_average:.4f} (average)"
            f" (Results verified on exported predictions JSON file.)"
        )

        return test_result_average
