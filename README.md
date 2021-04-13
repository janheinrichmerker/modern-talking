[![CI](https://img.shields.io/github/workflow/status/heinrichreimer/modern-talking/CI?style=flat-square)](https://github.com/heinrichreimer/modern-talking/actions?query=workflow%3A"CI")
[![Code coverage](https://img.shields.io/codecov/c/github/heinrichreimer/modern-talking?style=flat-square)](https://codecov.io/github/heinrichreimer/modern-talking/)
[![Issues](https://img.shields.io/github/issues/heinrichreimer/modern-talking?style=flat-square)](https://github.com/heinrichreimer/modern-talking/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/heinrichreimer/modern-talking?style=flat-square)](https://github.com/heinrichreimer/modern-talking/commits)
[![License](https://img.shields.io/github/license/heinrichreimer/modern-talking?style=flat-square)](LICENSE)

# 🗣️ modern-talking

Modern Talking: Key-Point Analysis using Modern Natural Language Processing

Participation at the [Quantitative Summarization – Key Point Analysis Shared Task](https://2021.argmining.org/shared_task_ibm.html#ibm) ([data on GitHub](https://github.com/ibm/KPA_2021_shared_task)).

## Usage

### Installation

First, install [Python 3](https://python.org/downloads/) and [Conda](https://docs.conda.io/en/latest/miniconda.html).
Then setup or update the Conda environment (may take a while):

```shell script
conda env update -f environment.yml
```

Now activate the Conda environment:

```shell script
conda activate modern-talking
```

### Testing

Run all unit tests:

```shell script
pytest
```

## License

This repository is licensed under the [MIT License](LICENSE).
