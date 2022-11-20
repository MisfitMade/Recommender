import os
import importlib.util
import pandas as pd
import numpy as np
from numpy.typing import NDArray

spec = importlib.util.spec_from_file_location("add", "../Recommender_Lib.py")
R_Lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R_Lib)

# if the path to the raw movie data does not exist, download all of it from google drive
R_Lib.verify_raw_data_exists()
R_Lib.verify_glove_data_exists()

# Tags strings
MOVIE_LENS_25M_TAG_COL = "tag"

# Misc DF strings
YEAR_COL = "year"

def accumulate_year(
    year_accumulator: list[str],
    title: str) -> str:

    toks = title.split()
    total_toks = len(toks)
    is_year = total_toks > 1
    if is_year:
        year = toks[total_toks-1]
        is_year = year.startswith('(') and year.endswith(')')
        if is_year:
            # then this is a year
            year_accumulator.append(year[1:-1])
            return " ".join(toks[:-1])

    # if made it here, found no year
    year_accumulator.append("0")
    return title


class SparseMatrixMaker:
    def __init__(self, bad_data_detector=(False, lambda _: False)):
        self.reset(bad_data_detector)

    def reset(self, bad_data_detector=(False, lambda _: False)) -> None:
        self.sparse_row_idxs = []
        self.sparse_col_idxs = []
        self.sparse_data = []
        self.sparse_col_count = 0
        self.sparse_row_count = 0
        self.is_bad_data = (bad_data_detector[0], np.vectorize(bad_data_detector[1]))
        print(
            "Sparse row count = ",
            self.sparse_row_count,
            " and sparse col count = ",
            self.sparse_col_count)


    def add_row_from_row_vec(
        self,
        row_vec_data: NDArray[np.float16],
        quiet: bool = False) -> None:
        '''
        Given a row vector of data, finds the col idxs of the data that is not 0,
        then using that and an array of the data that is not 0, add it as a
        row to the sparse mat maker. This method assumes that
        self.sparse_row_count is on the right row count.
        '''
        col_idxs, data = self.get_idxs_and_data(row_vec_data)
        self.add_row(col_idxs, data, quiet)


    def add_row(
        self,
        col_idxs: NDArray[np.int32],
        data: NDArray[np.float16],
        quiet: bool = False) -> None:
        '''
        Given col coords and data, adds a row to the sparse coords arrays
        then increments the row count. This method assumes that
        self.sparse_row_count is on the right row count.
        '''
        self.add_col_coords(col_idxs)
        row_coords = np.zeros(len(col_idxs))
        row_coords.fill(self.sparse_row_count)
        self.add_row_coords(row_coords)
        self.add_data(data)
        self.sparse_row_count = self.sparse_row_count + 1

        if not quiet or self.sparse_col_count % 1000 == 0:
            print(
                "Row coords len: ", len(self.sparse_row_idxs),
                "\nCol coords len: ", len(self.sparse_col_idxs),
                "\nData len: ", len(self.sparse_data))


    def add_column_from_col_vec(
        self,
        col_vec_data: NDArray[np.float16],
        quiet: bool = False) -> None:
        '''
        Given a col vector of data, finds the row idxs of the data that is not 0,
        then using that and an array of the data that is not 0, add it as a
        column to the sparse mat maker. This method assumes that
        self.sparse_row_count is on the right col count.
        '''
        row_idxs, data = self.get_idxs_and_data(col_vec_data)
        self.add_column(row_idxs, data, quiet)


    def add_column(
        self,
        row_idxs: NDArray[np.int32],
        data: NDArray[np.float16],
        quiet: bool = False) -> None:
        '''
        Given row coords and data, adds a column to the sparse coords arrays
        then increments the column count. This method assumes that
        self.sparse_col_count is on the right column count.
        '''
        self.add_row_coords(row_idxs)
        col_coords = np.zeros(len(row_idxs))
        col_coords.fill(self.sparse_col_count)
        self.add_col_coords(col_coords)
        self.add_data(data)
        self.sparse_col_count = self.sparse_col_count + 1

        if not quiet or self.sparse_col_count % 1000 == 0:
            print(
                "Row coords len: ", len(self.sparse_row_idxs),
                "\nCol coords len: ", len(self.sparse_col_idxs),
                "\nData len: ", len(self.sparse_data))
    

    def add_row_coords(self, row_coords: NDArray[np.float16]) -> None:
        '''
        Adds the given row coords to this sparse mat maker's row array
        '''
        self.sparse_row_idxs.extend(row_coords)


    def add_col_coords(self, col_coords: NDArray[np.float16]) -> None:
        '''
        Adds the given row coords to this sparse mat maker's row array
        '''
        self.sparse_col_idxs.extend(col_coords)


    def add_data(self, data: NDArray[np.float16]) -> None:
        '''
        Adds the given data to this sparse mat maker's data array
        '''
        self.sparse_data.extend(data)


    def get_idxs_and_data(
        self,
        data_vec: NDArray[np.float16]) -> tuple[NDArray[int], NDArray[np.float16]]:
        '''
        Finds the data that is not zero and the indexes they occur at
        '''
        idx_filter = data_vec != 0.0
        idxs = np.nonzero(idx_filter)[0]
        data = data_vec[idx_filter]
        if self.is_bad_data[0] and len(data) > 0:
            bads = self.is_bad_data[1](data)
            if np.any(bads):
                bad_idxs = np.nonzero(bads)[0]
                print("On row count ",
                    self.sparse_row_count,
                    " and col count ",
                    self.sparse_col_count,
                    " a bad value was detected in the following data\n",
                    data,
                    "\nAnd the bads are ",
                    data[bad_idxs],
                    "\nAnd the bad indices are ",
                    bad_idxs)                    

        return idxs, data


    def save_sparse_mat_coords(self, path_to_file_and_file: str):
        '''
        Saves this object's sparse coords as 3 arrays:
        row_coords, col_coords, data
        '''
        np.savez(
            path_to_file_and_file,
            sparse_coords_tag=np.array(
                [self.sparse_row_idxs, self.sparse_col_idxs]),
            sparse_data_tag=np.array(self.sparse_data))


    def make_movies_df_row_element_into_sparse_coords(
        self,
        movies_df_row: pd.Series,
        quiet: bool = False) -> int:
        '''
        Assumes a certain form of movies_df_row and converts the series
        into one long row's worth of data and adds it to the sparse mat coords
        '''
        row = []
        # know the element names, so hardcode it
        #print(movies_df_row)

        # Get the movie id, which is an int
        movie_id = movies_df_row.loc[R_Lib.MOVIE_LENS_25M_MOVIE_ID_COL]
        #print(movie_id)
        row.append(movie_id)

        # get the title embedding, which is a numpy array of floats
        title_mat = movies_df_row.loc[R_Lib.MOVIE_LENS_25M_MOVIE_TITLE_COL]
        flattened_title = title_mat.flatten()
        row.extend(flattened_title)
        
        # get the genres, which is a list of ints
        row.extend(movies_df_row.loc[R_Lib.MOVIE_LENS_25M_MOVIE_GENRES_COL])

        # get the year, which is a str
        row.append(int(movies_df_row.loc[YEAR_COL]))

        # get the tag matrrix which is a np array of floats
        tag_mat = movies_df_row.loc[MOVIE_LENS_25M_TAG_COL]
        flattened_tag_mat = tag_mat.flatten()
        row.extend(flattened_tag_mat)            

        # add this as a row to the sparse mat
        row = np.array(row).astype(np.float32)
        self.add_row_from_row_vec(row, quiet)

        return movie_id

    def store_ratings_for_mov_id_as_sparse_coords(
        self,
        movie_id: int,
        ratings_df: pd.DataFrame,
        reindexer: NDArray[int],
        quiet: bool = False) -> int:
        '''
        Given a movie ID and a ratings DF, makes a column of ratings for that movie_id for
        all users where it has the rating for the users who have rated the movie, else 0.
        It then saves this info as a row, col, data coordinate for sparse matrix interpretation,
        to the global lists that hold each:
        sparse_row_idxs
        sparse_col_idxs
        sparse_data

        This takes a long while to do all the movies, so it can report progress to avoid seeming stalled.
        '''
        movie_id_col = str(movie_id)
        # get user IDs and their rating for users who rated the movie with movie_id
        df = ratings_df[ratings_df[R_Lib.MOVIE_LENS_25M_MOVIE_ID_COL] == movie_id][[R_Lib.MOVIE_LENS_25M_USER_ID_COL, R_Lib.MOVIE_LENS_25M_RATING_COL]]
        df.columns = [R_Lib.MOVIE_LENS_25M_USER_ID_COL, movie_id_col]
        # make the user IDs be the row index
        df.set_index(R_Lib.MOVIE_LENS_25M_USER_ID_COL, inplace=True)
        # insert a row with a 0 for every user ID who did not rate movie with movie_id
        df = df.reindex(reindexer, fill_value='0')

        # save this info to the sparse data info
        col_vec = df[movie_id_col].to_numpy().astype(np.float16)
        self.add_column_from_col_vec(col_vec, quiet)

        return movie_id


'''
*** SCRATCH ***
___________________DEBUG MAKING RATINGS MATRIX_____________________________
movie_id_col = str(235)
# get user IDs and their rating for users who rated the movie with movie_id
df = ratings_df[ratings_df[MOVIE_LENS_25M_MOVIE_ID_COL] == 235][[MOVIE_LENS_25M_USER_ID_COL, MOVIE_LENS_25M_RATING_COL]]
df.columns = [MOVIE_LENS_25M_USER_ID_COL, movie_id_col]
# make the user IDs be the row index
df.set_index(MOVIE_LENS_25M_USER_ID_COL, inplace=True)
# insert a row with a 0 for every user ID who did not rate movie with movie_id
df = df.reindex(reindexer, fill_value='0')

q = df[movie_id_col].to_numpy().astype(np.float16)

idx_filter = q != 0.0
idxs = np.nonzero(idx_filter)[0]

q

idx_filter

idxs

d = q[idx_filter]

d

#[3 for _ in range(len(idxs))]
# np.full((1,len(idxs)), 3)[0]
z = np.zeros(len(idxs))

z.fill(3)
z
----------------------------------------------------------------------------------
'''