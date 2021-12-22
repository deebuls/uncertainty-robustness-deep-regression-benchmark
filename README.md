# Robust Maximum Likelihood Uncertainty Estimation for Deep Regression
*SafeAI 2022(at AAAI-22) Submission*

This repository contains the code to reproduce all results presented in the SafeAI submission: "Maximum Likelihood Uncertainty Estimation: Robustness to Outliers".

## The repository is an extension to the code submitted by "Deep Evidential Regression" paper in Neurips 2021. 


## Setup

### Download datasets
To get started, first download the relevant datasets and some pre-trained models (if you don't want to re-train from scratch). These datasets include:
1. [NYU Depth v2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html "NYU Depth v2 dataset ")
2. [Apolloscapes depth](http://apolloscape.auto/stereo.html "Apolloscapes depth")
3. UCI regression tasks (already pre-downloaded in `./data`)

To download run the following commands from a Unix shell:
```
cd evidential_regression
bash ./download_data.sh
```
We also include pre-trained models, in case reviewers would like to use these to produce results without retraining from scratch. If reviewers would like to re-train, we provide the code to do so - we include pre-trained models here only for added convenience. The provided datasets are also preprocessed, to use the raw datasets our preprocessing scripts can be found in `./preprocess/*`.


### Software environment
We package our codebase into a conda environment, with all dependencies listed in `environment.yml`. To create a local copy of this environment and activate it, run the following commands:
```
conda env create -f environment.yml
conda activate evidence
```


## Reproducing Results

### Monocular Depth
The easiest way to reproduce the depth results presented in the submission would be to run:
```
python gen_depth_results.py
```
This command will automatically use the pre-trained models downloaded above, if you would like to retrain the depth models from scratch you can run:
```
python train_depth.py  [--model {evidential, dropout, ensemble}]
```
Note that the path to any new trained models should be replaced in the `trained_models` parameter within `gen_depth_results.py` if you'd like the plots to reflect changes in this newly trained model.


### UCI and Cubic Examples
Results for the cubic example figures can be reproduced by running:
```
python run_cubic_tests.py
```

Results for the UCI benchmarking tasks can be reproduced by running:
```
python run_uci_dataset_tests.py [-h] [--num-trials NUM_TRIALS]
                                [--num-epochs NUM_EPOCHS]
                                [--datasets {yacht, ...}]]
```
Thank you!
