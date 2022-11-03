# CS 567 Recommender System

## Example of setting up a conda environment for this project
#### Maybe not neccesary, but guarantees you'll be able to use, debug, run the code + keep you from creating a clashing Python version dependency between Python projects. Have conda installed, then in a conda terminal:
* `conda create -n Recommender python=3.10`
* `conda activate Recommender`
* `pip install tensorflow datasets jupyterlab numpy matplotlib pandas scikit-learn`

Now, when you work in the project and/or run its code, do so in this Recommender environment/conda space.

If you are attempting to train a tensorflow NN and are getting a warning about the work not being
mapped to the GPU because the library `libcudnn8` is not installed, then you can install it via conda:
* `conda install -c anaconda cudnn`

## Files
* `Recommender_Training.ipynb` -- This is a notebook used to prep data for training, train, save and analyze results.
* `Recommender_Classification.ipynb` -- This is a notebook used classify with saved model settings.
* `DataAugmentation.ipynb` -- This is a notebook used to gather, assemble and create data for training.
* `Recommender_Lib.py` -- This is a python library of dev defined functions.

## General setup before running any code

## Using the Recommender code

