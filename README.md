# CS 567 Recommender System

## Example of setting up a conda environment for this project
#### Maybe not neccesary, but guarantees you'll be able to use, debug, run the code + keep you from creating a clashing Python version dependency between Python projects. Have conda installed, then in a conda terminal:
* `conda create -n Recommender python=3.10`
* `conda activate Recommender`
* `pip install tensorflow tensorflow_hub jupyterlab numpy matplotlib pandas scikit-learn gdown`


## Files
* `Recommender_Construction.ipynb` -- This is a notebook used to prep data for training, train, save and analyze results.
* `Recommender_Classification.ipynb` -- This is a notebook used to offer movie recommendations with saved models and their settings.
* `resources/DataAugmentation.ipynb` -- This is a notebook used to gather, assemble and create data for training.
* `Recommender_Lib.py` -- This is a python library of dev defined functions and state used by all other files.
* `resources/DataAugmenttion_Lib.py` -- This is a python library of dev defined functions and state used by `DataAugmentation.ipynb`

## General setup before running any code
-- None. All necessary data is fetched at runtime if it does not exist via download from google drive.
-- Note: There is alot of data here during run time. If using the notebooks, you will probably have to reset your kernel between using `resources/DataAugmentation.ipynb` to `Recommender_Construction.ipynb` to `Recommender_Classification.ipynb` so as to dump the jupyter notebook states saved in RAM.

## Using the Recommender code
* You can find some nice comments and explainations throughout the code, in both the notebook files and amongst the functions in DataAugmentation_Lib.py and Recommender_Lib.py
* The file `resources/DataAugmentation.ipynb` has the work flow for encoding and embedding the raw .dat and .csv files of movie, user and ratings data into entirely numeric matrices along with comments explaining how to use it/what it does. It does some inspecton of the data such as use age and occupation, using tables and plots. It makes an approximately (162,000 users x 65,000 movies) sparse ratings matrix. There is cleaning, combining, processing, building, encoding, embedding. Certain specs, such as sizes, token embedding counts, will be saved to a json file and polled later to ensure correctness across the notebooks. Some of the cells can take a long time to complete and most likely have a comment in it saying so if that is the case.
* The file `Recommender_Construction.ipynb` has the work flow for taking the processed data and making a cosine similairity matrix (huge) for the movies and a kmeans clustered model of the users after embedding them to be used later during making of movie recommendations, along with comments explaining how to use it/what it does. There is building, encoding, embedding. It does some inspecton of the users embedding matrix, as well as the best k for kmeans clustering, and clustering, via plots. Certain specs, such as sizes, token embedding counts, will be saved to a json file and polled later to ensure correctness across the notebooks. Some of the cells can take a long time to complete and most likely has a comment in it saying so if that is the case.
* The file `Recommender_Classification.ipynb` has the workflow for defining users for which you want to recommend movies for, then using the cosine similairity matrix, the sparse ratings matrix, and embedding the users then using the saved-and-loaded in kmeans clustered model; each cell includes comments explaining how to use it/what it does. This file will make 3 kinds of recommendations. Movies we recommend based off them being
  * cosine similar to the movie the user says they like.
  * movies most liked by other users who like the movie the user likes.
  * movies most liked by other users similar to the user in embedded characteristics: gender, age, occupation, zipcode.
* The files `resources/DataAugmenttion_Lib.py` and `Recommender_Lib.py` contain function and object definitions used throughout the entire process: augment, clean, encode, emmbed, cluster, leverage and predict data.
