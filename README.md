[![DOI](https://zenodo.org/badge/694175632.svg)](https://zenodo.org/doi/10.5281/zenodo.13117607)
# SMoLK
This repository contains the notebook responsible for training and testing the models in our paper, [Learned Kernels for Sparse, Interpretable and Efficient PPG Signal Quality Assessment and Artifact Segmentation](https://arxiv.org/abs/2307.05385).

We have laid out a Jupyter Notebook containing the code and comments in chronological order, allowing one to train, test, and prune Learned Kernel Models as they appear in the paper. Please refer to the main Jupyter Notebook in this repository for the instructions and explanations surrounding this code — rather than providing a long readme to refer back to, we thought it would be easier to have a single blog-like guide from which one can walk through the experiments. Thank you!

## Requirements
To run this code, you will need the following packages (version more recent than what is listed will likely work):
```
torch==2.1.1
scikit-learn==1.3.2
scipy==1.9.3
numpy==1.22.4
tqdm==4.66.1
```
Depending on your internet speed, this should only take around 20 minutes to install, though your times may vary depending on your machine. Machine-specific installation of PyTorch is recommended [here](https://pytorch.org/get-started/locally/)

## Generating the Datasets
The Zheng et al. dataset can be found at [this PhysioNet Page](https://physionet.org/content/ecg-arrhythmia/1.0.0/), and the Computing in Cardiology 2017 dataset can be found at [this Physionet Page](https://physionet.org/content/challenge-2017/1.0.0/).

First, create two new directories in the base Git repository, named `CinC` and `Zheng et al`. Next, unzip `training2017.zip` found in the Computing in Cardiology 2017 dataset at the above link. Move the contents of this unzipped file (it should be a single folder named `training2017`) into the `CinC` directory created earlier.

For the Zheng et al. dataset, copy `ConditionNames_SNOMED-CT.csv` into `Zheng et al`, then use the `utils/preprocess_zhengetal.ipynb` notebook to generate the `preprocessed_data.pkl` file for this dataset. In the notebook, `base_dir` should point to the `WFDBRecords` directory in the downloaded PhysioNet dataset. After `preprocessed_data.pkl` is generated, move it to `Zheng et al`. If done properly, `CinC` should contain one directory, `training2017`, and `Zheng et al` should contain two files, `ConditionNames_SNOMED-CT.csv` and `preprocessed_data.pkl`. Both `training2017` and `Zheng et al` should be at the base of the SMoLK directory.

The PPG dataset is included with this repository, thus no other steps need to be taken in this regard.
