# kubeflow
Minimal examples using Kubeflow

## Spotlight implicit sequence model pipeline
The baseline example is a Kubeflow pipeline for a [Spotlight](https://maciejkula.github.io/spotlight/index.html#) implicit sequence model.

### `pipeline.py`
Defines a Kubeflow pipeline function chaining sequential data generation and the training of a Spotlight implicit sequence model, and compiles into workflow yaml.

### `preprocess_sequential.py`
Creates a synthetic sequential interactions dataset, generates a train/test split with a fixed seed, and persists the train and test sequences datasets to disk.

### `train.py`
Reads a train sequences dataset from disk, and trains a Spotlight implicit sequence model for a fixed number of iterations, saving the Pytorch model object to disk.

### `utils.py`
Exposes a minimal util for writing and reading Spotlight [SequenceInteractions](https://maciejkula.github.io/spotlight/_modules/spotlight/interactions.html) from disk, as well as logging utilities.

## Dockerfiles
The Dockerfiles `preprocess_sequential.Dockerfile` and `train.Dockerfile` define the environments in which the preprocessing and training ops in the Kubeflow pipeline are run. Both build simple Conda environments as stipulated by `environment.yml`. A base requirements image should probably be built to streamline the process and avoid repetition.
