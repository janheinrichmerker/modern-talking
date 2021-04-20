from pathlib import Path
from urllib.request import urlretrieve

urls = [
    "https://github.com/IBM/KPA_2021_shared_task/raw/main/kpm_data/arguments_dev.csv",
    "https://github.com/IBM/KPA_2021_shared_task/raw/main/kpm_data/arguments_train.csv",
    "https://github.com/IBM/KPA_2021_shared_task/raw/main/kpm_data/key_points_dev.csv",
    "https://github.com/IBM/KPA_2021_shared_task/raw/main/kpm_data/key_points_train.csv",
    "https://github.com/IBM/KPA_2021_shared_task/raw/main/kpm_data/labels_dev.csv",
    "https://github.com/IBM/KPA_2021_shared_task/raw/main/kpm_data/labels_train.csv",
]


def filename(url):
    return url.rsplit('/', 1)[1]


data_dir = Path(__file__).parent.parent / "data"
for url in urls:
    name = filename(url)
    file = data_dir / name
    if not file.exists():
        print("Download {} from {}.".format(name, url))
        urlretrieve(url, file)
    else:
        print("File {} already downloaded.".format(name))
print("Data downloaded.")