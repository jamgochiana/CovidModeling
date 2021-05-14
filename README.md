# CovidModeling
Covid-19 Modeling Repo for Convex Optimization II

## File stucture
- `src` contains the source code for the package
- `datasets` contains datasets pertaining to experiments.
	- `datasets/raw` contains raw datasets
	- `datasets/processed` contains processed datasets
	- `datasets/process_utils` contains utilities for processing datasets
- `experiments` contains any experiments conducted in the submitted manuscript.

## Environment
A conda environment for exact experiment reproduction can be set up and uses with:
```
conda create -y --name covid python==3.7
conda install -y --name covid --file requirements.txt -c conda-forge -c pytorch
conda activate covid
...
conda deactivate
```

## Download and Process Datasets
Run `python datasets/download_and_process_data.py` to download and process datasets.

## Running Experiments
To run experiments from the associated paper:

1. Install and activate the environment.
2. Download and process the datasets.
3. Run `experiments/main.py` to run experiments
