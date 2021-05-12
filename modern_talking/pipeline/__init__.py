from csv import reader
from pathlib import Path
from typing import Set

from modern_talking.evaluation import Metric
from modern_talking.matchers import Matcher
from modern_talking.model import Argument, KeyPoint, Labels, LabelledDataset, \
    DatasetType


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

        data_dir = Path(__file__).parent.parent.parent / "data"

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
            csv = reader(file)
            return {
                Argument(row[0], row[1], row[2], int(row[3]))
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
            csv = reader(file)
            return {
                KeyPoint(row[0], row[1], row[2], int(row[3]))
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
            csv = reader(file)
            return {
                (row[0], row[1]): row[2]
                for row in csv
            }

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

        # Load datasets
        train_data = Pipeline.load_dataset(DatasetType.TRAIN)
        dev_data = Pipeline.load_dataset(DatasetType.DEV)
        test_data = Pipeline.load_dataset(DatasetType.TEST) \
            if not ignore_test else dev_data

        # Train model.
        self.matcher.train(train_data, dev_data)
        # Predict labels for test data.
        predicted_labels = self.matcher.predict(test_data)
        # Get ground-truth labels from test data.
        ground_truth_labels = test_data.labels
        # Evaluate labels.
        return self.metric.evaluate(predicted_labels, ground_truth_labels)
