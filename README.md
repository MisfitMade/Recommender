# CS 567 Recommender System

## Example of setting up a conda environment for this project
#### Maybe not neccesary, but guarantees you'll be able to use, debug, run the code + keep you from creating a clashing Python version dependency between Python projects. Have conda installed, then in a conda terminal:
* `conda create -n Recommender python=3.10`
* `conda activate Recommender`
* `pip install tensorflow tensorflow_hub jupyterlab numpy matplotlib pandas scikit-learn gdown`


## Files
* `Recommender_Construction.ipynb` -- This is a notebook used to prep data for training, train, save and analyze results.
* `Recommender_Classification.ipynb` -- This is a notebook used classify with saved model settings.
* `DataAugmentation.ipynb` -- This is a notebook used to gather, assemble and create data for training.
* `Recommender_Lib.py` -- This is a python library of dev defined functions and state used by all other files.
* `DataAugmenttion_Lib.py` -- This is a python library of dev defined functions and state used by `DataAugmentation.ipynb`
* `DataAugScript.py` -- This is just a python file used to carry out some of the tasks at the `DataAugmentation.ipynb` file so as to not tie up the notebook kernel during the longer tasks.

## General setup before running any code

## Using the Recommender code

