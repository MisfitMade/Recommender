# These importing lines pull in the Recommender_Lib from folders above this file
from DataAugmentation_Lib import *

# These importing lines pull in env installed libs
import json
import pandas as pd
import numpy as np

'''
Read each .dat file into a DF and give them col headers to match the .csv 25 mil data sets
'''
ratings_1M_df = pd.read_table(
    PATH_TO_MOVIE_LENS_1M_RATINGS,
    sep=MOVIE_LENS_1M_DELIM, # sep="::"
    header=None, # tell the read that the data has no col headers
    engine="python") # explicit use of python engine to use > 1 length sep ("::") without a warning
ratings_1M_df.columns = MOVIE_LENS_25M_RATINGS_COLS

ratings_25M_df = pd.read_csv(PATH_TO_MOVIE_LENS_25M_RATINGS)

users_df = pd.read_table(
    PATH_TO_MOVIE_LENS_1M_USERS,
    sep=MOVIE_LENS_1M_DELIM,
    header=None,
    engine="python")
users_df.columns = MOVIE_LENS_1M_USERS_COLS

movies_1M_df = pd.read_table(
    PATH_TO_MOVIE_LENS_1M_MOVIES,
    sep=MOVIE_LENS_1M_DELIM,
    header=None,
    engine="python",
    encoding="ISO-8859-1") # default utf8 encoding fails during read of movies data.
movies_1M_df.columns = MOVIE_LENS_25M_MOVIES_COLS

movies_25M_df = pd.read_csv(PATH_TO_MOVIE_LENS_25M_MOVIES)

'''
Combine the movies DFs. Need to change the 1M movie IDs so we can tell them apart later by
adding the highest movie ID in the 25M to each. Then concat the movies in the 1M DF that are
not in the 25M DF to the end of the 25M Df
'''
max_25M_movie_id = movies_25M_df[MOVIE_LENS_25M_MOVIE_ID_COL].max()
add_max_movie_id = lambda id: id + max_25M_movie_id
movies_1M_df[MOVIE_LENS_25M_MOVIE_ID_COL] = movies_1M_df[MOVIE_LENS_25M_MOVIE_ID_COL].map(
    add_max_movie_id)
ratings_1M_df[MOVIE_LENS_25M_MOVIE_ID_COL] = ratings_1M_df[MOVIE_LENS_25M_MOVIE_ID_COL].map(
    add_max_movie_id)

movies_df = pd.concat(
    [movies_25M_df,
    movies_1M_df[~movies_1M_df[MOVIE_LENS_25M_MOVIE_TITLE_COL].isin(
        movies_25M_df[MOVIE_LENS_25M_MOVIE_TITLE_COL])]],
    ignore_index=True)

'''
Now need to change the 1M user IDs in all DFs that have userIds so that I can tell them apart
from the 25M user IDs. Then combine them and fill in the blanks with 0s.
'''
total_25M_users = len(ratings_25M_df[MOVIE_LENS_25M_USER_ID_COL].unique())
add_max_user_id = lambda x: x + total_25M_users
users_df[MOVIE_LENS_25M_USER_ID_COL] = users_df[MOVIE_LENS_25M_USER_ID_COL].map(
    add_max_user_id)
ratings_1M_df[MOVIE_LENS_25M_USER_ID_COL] = ratings_1M_df[MOVIE_LENS_25M_USER_ID_COL].map(
    add_max_user_id)

users_df = pd.concat([pd.DataFrame({
    MOVIE_LENS_25M_USER_ID_COL : ratings_25M_df[MOVIE_LENS_25M_USER_ID_COL].unique()
    }),
    users_df],
    ignore_index=True).fillna(0)
ratings_df = pd.concat([ratings_25M_df, ratings_1M_df], ignore_index=True)    

'''
Define some variables that will help us load stuff later and make decisions below
'''
total_users = users_df.shape[0]
total_movies = movies_df.shape[0]
reindexer = np.arange(1, total_users+1)
'''
Give a function to the SparseMatrixMaker to check for bad data so that if the ratings
data matrix gets outof the range of the ratings numbers, 0.0 - 5.0, then an exception
will throw, because something is broken.
'''
sparse_mat_maker = SparseMatrixMaker(bad_data_detector=(True, lambda d: d < 0.0 or d > 5.0))
print("Total users: ", total_users, "\nTotal movies: ", total_movies)

'''
Now we have 3 DFs, two of which correspond to user ratings: users_df, ratings_df. Lets combine them.
We have movies_df, which has all the movie IDs, so make a column for each movie ID to users_df
and include the rating if the user has rated it.
It takes awhile to run, longer if you give it a quiet=False so that it reports progress
to not appear stlled. After, save the sparse coords and data as a numpy.
'''
# The store_ratings_for_mov_id_as_sparse_coords method builds a sparse matrix
# coordinates into 3 long arrays and those are saved to be loaded in later.
# store_ratings_for_mov_id_as_sparse_coords just returns the movie_id it is given.
movie_ids_series = movies_df[MOVIE_LENS_25M_MOVIE_ID_COL].map(
    lambda movie_id: sparse_mat_maker.store_ratings_for_mov_id_as_sparse_coords(
        movie_id,
        ratings_df,
        reindexer,
        quiet=True))

sparse_mat_maker.save_sparse_mat_coords(PATH_TO_PROCESSED_RATINGS_DATA)
print("Finshed and saved ratings matrix.")
