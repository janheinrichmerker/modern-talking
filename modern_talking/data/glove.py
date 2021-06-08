from os.path import splitext
from pathlib import Path
from typing import List
from urllib.request import urlretrieve
from zipfile import ZipFile

from numpy import fromstring, zeros, ndarray

from modern_talking.data import filename

tokens = 42
dimensions = 300
url = f"https://nlp.stanford.edu/data/glove.{tokens}B.{dimensions}d.zip"

data_dir = Path(__file__).parent.parent.parent / "data"
glove_dir = data_dir
name: str = filename(url)
glove_zip = glove_dir / name
glove_txt_name = splitext(name)[0] + ".txt"
glove_file = glove_dir / glove_txt_name


def get_glove_embedding_matrix(voc: List[str]) -> ndarray:
    embeddings_index = {}
    with glove_file.open("r") as file:
        for line in file:
            word, coefficients = line.split(maxsplit=1)
            coefficients = fromstring(coefficients, "f", sep=" ")
            embeddings_index[word] = coefficients
    print(f"Found {len(embeddings_index)} word vectors.")

    hits = 0
    misses = 0

    embedding_matrix: ndarray = zeros((len(voc) + 2, dimensions))
    for i, word in enumerate(voc):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f"Converted {hits} words ({misses} misses).")
    return embedding_matrix


def download_glove_embeddings() -> None:
    if not glove_file.exists():
        if not glove_zip.exists():
            print(f"Download GloVe embeddings from {url} to {glove_zip}.")
            urlretrieve(url, glove_zip)
            print("GloVe embeddings downloaded.")
        else:
            print(f"GloVe embeddings already downloaded to {glove_zip}.")
        print(f"Unzip GloVe embeddings from {glove_zip} to {glove_file}.")
        with ZipFile(glove_zip, 'r') as zip_file:
            zip_file.extractall(glove_dir, [glove_txt_name])
        print("GloVe embeddings unzipped.")
    else:
        print(f"GloVe embeddings already unzipped to {glove_file}.")
