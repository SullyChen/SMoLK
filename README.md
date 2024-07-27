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
