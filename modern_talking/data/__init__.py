from pathlib import Path
from urllib.request import urlretrieve

paths = [
    "kpm_data/arguments_dev.csv",
    "kpm_data/arguments_train.csv",
    "test_data/arguments_test.csv",
    "kpm_data/key_points_dev.csv",
    "kpm_data/key_points_train.csv",
    "test_data/key_points_test.csv",
    "kpm_data/labels_dev.csv",
    "kpm_data/labels_train.csv",
]
urls = [
    f"https://github.com/IBM/KPA_2021_shared_task/raw/main/{path}"
    for path in paths
]


def filename(url):
    return url.rsplit('/', 1)[1]


def download_kpa_2021_data() -> None:
    data_dir = Path(__file__).parent.parent.parent / "data"
    all_datasets_exists = all(
        (data_dir / filename(url)).exists()
        for url in urls
    )
    if all_datasets_exists:
        print("Datasets already downloaded.")
        return
    else:
        print("Download KPA 2021 datasets.")

    for url in urls:
        name = filename(url)
        file = data_dir / name
        if not file.exists():
            print("Download {} from {}.".format(name, url))
            urlretrieve(url, file)
        else:
            print("File {} already downloaded.".format(name))
    print("Datasets downloaded.")
