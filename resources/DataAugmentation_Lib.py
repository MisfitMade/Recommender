import os
import gdown
import importlib.util
import pandas as pd
import numpy as np
from numpy.typing import NDArray

spec = importlib.util.spec_from_file_location("add", "../Recommender_Lib.py")
R_Lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R_Lib)

USERS_MATRIX = R_Lib.USERS_MATRIX
PATH_TO_PROCESSED_USERS_DATA = R_Lib.PATH_TO_PROCESSED_USERS_DATA
PATH_TO_PROCESSED_RATINGS_DATA = R_Lib.PATH_TO_PROCESSED_RATINGS_DATA
PATH_TO_PROCESSED_MOVIES_DATA = R_Lib.PATH_TO_PROCESSED_MOVIES_DATA
PATH_TO_PROCESSED_DATA_SPECS = R_Lib.PATH_TO_PROCESSED_DATA_SPECS
SPEC_MOVIE_IDS = R_Lib.SPEC_MOVIE_IDS
SPEC_USER_IDS = R_Lib.SPEC_USER_IDS
SPEC_MOVIES_MATRIX_COLUMN_COUNT = R_Lib.SPEC_MOVIES_MATRIX_COLUMN_COUNT

PATH_TO_MOVIE_LENS = os.path.join(R_Lib.PATH_TO_RESOURCES, "MovieLens")
PATH_TO_MOVIE_LENS_1M = os.path.join(PATH_TO_MOVIE_LENS, "1Mill")
PATH_TO_MOVIE_LENS_25M = os.path.join(PATH_TO_MOVIE_LENS, "25Mill")
PATH_TO_MOVIE_LENS_1M_RATINGS = os.path.join(PATH_TO_MOVIE_LENS_1M, "ratings.dat")
PATH_TO_MOVIE_LENS_25M_RATINGS = os.path.join(PATH_TO_MOVIE_LENS_25M, "ratings.csv")
PATH_TO_MOVIE_LENS_1M_USERS = os.path.join(PATH_TO_MOVIE_LENS_1M, "users.dat")
PATH_TO_MOVIE_LENS_1M_MOVIES = os.path.join(PATH_TO_MOVIE_LENS_1M, "movies.dat")
PATH_TO_MOVIE_LENS_25M_MOVIES = os.path.join(PATH_TO_MOVIE_LENS_25M, "movies.csv")
PATH_TO_MOVIE_LENS_25M_TAGS = os.path.join(PATH_TO_MOVIE_LENS_25M, "tags.csv")
# if the path to the raw movie data does not exist, download all of it from google drive
raw_data_exists = lambda: (os.path.exists(PATH_TO_MOVIE_LENS)
    and os.path.exists(PATH_TO_MOVIE_LENS_1M)
    and os.path.exists(PATH_TO_MOVIE_LENS_25M)
    and os.path.exists(PATH_TO_MOVIE_LENS_1M_RATINGS)
    and os.path.exists(PATH_TO_MOVIE_LENS_25M_RATINGS)
    and os.path.exists(PATH_TO_MOVIE_LENS_1M_USERS)
    and os.path.exists(PATH_TO_MOVIE_LENS_1M_MOVIES)
    and os.path.exists(PATH_TO_MOVIE_LENS_25M_MOVIES)
    and os.path.exists(PATH_TO_MOVIE_LENS_25M_TAGS))
if not raw_data_exists():
    print("Downloading necessary raw data.")
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/1KNYIV7wmIHAFeoc41VIiX7yt7-FlgZfF?usp=sharing",
        quiet=False)
    # make sure it worked
    if not raw_data_exists():
        raise Exception(f"Could not download raw data. Are you connected to the internet?")

PATH_TO_GLOVE = os.path.join(R_Lib.PATH_TO_RESOURCES, "Glove_twitter_vers")
# if the path to the glove components do not exist, download them from google drive
glove_data_exists = lambda: (os.path.exists(PATH_TO_GLOVE)
    and os.path.exists(os.path.join(PATH_TO_GLOVE, "25d.txt"))
    and os.path.exists(os.path.join(PATH_TO_GLOVE, "50d.txt"))
    and os.path.exists(os.path.join(PATH_TO_GLOVE, "100d.txt"))
    and os.path.exists(os.path.join(PATH_TO_GLOVE, "200d.txt")))
if not glove_data_exists():
    print("Downloading necessary Glove components.")
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/1iddQ-LoQ-ynEK-OjpDJJiCvBtBfWuYql?usp=sharing",
        quiet=False)
    # make sure it worked
    if not glove_data_exists():
        raise Exception(f"Could not download glove components. Are you connected to the internet?")

MOVIE_LENS_1M_DELIM = "::"

# Ratings strings
MOVIE_LENS_25M_USER_ID_COL = "userId"
MOVIE_LENS_25M_MOVIE_ID_COL = "movieId"
MOVIE_LENS_25M_RATING_COL = "rating"
MOVIE_LENS_25M_TIMESTAMP_COL = "timestamp"
MOVIE_LENS_25M_RATINGS_COLS = [
    MOVIE_LENS_25M_USER_ID_COL,
    MOVIE_LENS_25M_MOVIE_ID_COL,
    MOVIE_LENS_25M_RATING_COL,
    MOVIE_LENS_25M_TIMESTAMP_COL]
# Users strings
MOVIE_LENS_1M_GENDER_COL = "gender"
MOVIE_LENS_1M_AGE_COL = "age"
MOVIE_LENS_1M_OCCUPATION_COL = "occupation"
MOVIE_LENS_1M_ZIPCODE_COL = "zip-code"
MOVIE_LENS_1M_USERS_COLS = [
    MOVIE_LENS_25M_USER_ID_COL,
    MOVIE_LENS_1M_GENDER_COL,
    MOVIE_LENS_1M_AGE_COL,
    MOVIE_LENS_1M_OCCUPATION_COL,
    MOVIE_LENS_1M_ZIPCODE_COL]
# Movies strings
MOVIE_LENS_25M_MOVIE_ID_COL = "movieId"
MOVIE_LENS_25M_MOVIE_TITLE_COL = "title"
MOVIE_LENS_25M_MOVIE_GENRES_COL = "genres"
MOVIE_LENS_25M_MOVIES_COLS = [
    MOVIE_LENS_25M_MOVIE_ID_COL,
    MOVIE_LENS_25M_MOVIE_TITLE_COL,
    MOVIE_LENS_25M_MOVIE_GENRES_COL]
ACTION = "Action"
ADVENTURE = "Adventure"
ANIMATION = "Animation"
CHILDRENS = "Children's"
COMEDY = "Comedy"
CRIME = "Crime"
DOCUMENTARY = "Documentary"
DRAMA = "Drama"
FANTASY = "Fantasy"
FILM_NOIR = "Film-Noir"
HORROR = "Horror"
MUSICAL = "Musical"
MYSTERY = "Mystery"
ROMANCE = "Romance"
SCI_FI = "Sci-Fi"
THRILLER = "Thriller"
WAR = "War"
WESTERN = "Western"
MOVIE_LENS_GENRES = [ACTION, ADVENTURE, ANIMATION, CHILDRENS, COMEDY, CRIME, DOCUMENTARY, DRAMA,
    FANTASY, FILM_NOIR, HORROR, MUSICAL, MYSTERY, ROMANCE, SCI_FI, THRILLER, WAR, WESTERN]
MOVIE_LENS_NUM_GENRES = len(MOVIE_LENS_GENRES)
# Tags strings
MOVIE_LENS_25M_TAG_COL = "tag"

# Misc DF strings
YEAR_COL = "year"

class Glove:
    def __init__(self, path_to_glove_folder: str, embedded_vector_len: int) -> None:
        
        self.id_to_embedding_map = {}
        self.word_to_vec_mapping = {}
        self.embedded_vec_len = embedded_vector_len
        path = os.path.join(path_to_glove_folder, f"{self.embedded_vec_len}d.txt")
        if os.path.isfile(path):
            with open(path, "r", encoding = "utf-8") as glove_f:
                for line in glove_f:
                    toks = line.split()
                    self.word_to_vec_mapping[toks[0]] = np.array(toks[1:])
        else:
            raise Exception(f"Glove file not found: {path}")

        self.bad_chars =  ['(', ')', ',', '\"', '!', '?', '.', ';', '*']
        self.bad_toks = ['-', "a", "the", "on", "of"]
        self.connecting_chars = ['_', '-', ':']
        '''
        I think the lowest movie year is 1874
        '''
        self.okay_connected_words = [
            "sci-fi",
            "zeta-jones",
            "coca-cola",
            "post-apocalyptic",
            "re-watch"]
        self.tok_mapper = {
            '9/11': "nine eleven",
            '%': " percent",
            'w/': "with",
            '007': "James Bond",
            '1st': "first",
            '2nd': "second",
            '3rd': "third",
            '4th': "fourth",
            '5th': "fifth",
            '6th': "sixth",
            '7th': "seventh",
            '8th': "eighth",
            '9th': "nineth",
            '10th': "tenth",
            '11th': "eleventh",
            '12th': "twelfth",
            '13th': "thirteenth",
            '14th': "fourteenth",
            '15th': "fifteenth",
            '16th': "sixteenth",
            '17th': "seventeenth",
            '18th': "eighteenth",
            '19th': "nineteenth",
            '20th': "twentieth",
            '21st': "twenty first",
            '0': "zero",
            '1': "one",
            '2': "two",
            '3': "three",
            '4': "four",
            '5': "five",
            '6': "six",
            '7': "seven",
            '8': "eight",
            '9': "nine",
            '10': "ten",
            '11': "eleven",
            '70': "seventy",
            '250': "two fifty",
            '20s': "twenties",
            '30s': "thirties",
            '40s': "fourties",
            '50s': "fifties",
            '60s': "sixties",
            '70s': "seventies",
            '80s': "eighties",
            '90s': "nineties",
            '1300s': "thirteen hundreds",
            '1600s': "sixteen hundreds",
            '1700s': "seventeen hundreds",
            '1790s': "seventeen nineties",
            '1800s': "eighteen hundreds",
            '1820s': "eighteeen twenties",
            '1840s': "eighteeen fourties",
            '1860s': "eighteeen sixties",
            '1870s': "eighteeen seventies",
            '1890s': "eighteeen nineties",
            '1900s': "nineteen hundreds",
            '1910s': "nineteen tens",
            '1920s': "nineteen twenties",
            '1930s': "nineteen thirties",
            '1940s': "nineteen fourties",
            '1950s': "nineteen fifties",
            '1960s': "nineteen sixties",
            '1970s': "nineteen seventies",
            '1980s': "nineteen eighties",
            '1990s': "nineteen nineties",
            '2000s': "two thousands",
            '2010s': "twenty tens",
            '2020s': "twenty twenties",
        }
        self.mapper_keys = self.tok_mapper.keys()

    def clean_str(self, string: str) -> str:
        ''''
        Cleans the string given. Does not cover all the cases seen in the tags.csv,
        but does cover some, creating some more words that Glove will recognize.
        '''

        dirty_toks = str(string).split()
        clean_toks = []
        for tok in dirty_toks:
            tok = tok.lower()
            if tok in self.bad_toks:
                continue # skip it

            for bad_char in self.bad_chars:
                if bad_char in tok:
                    tok = tok.replace(bad_char, '')
            
            if tok in self.mapper_keys:
                tok = self.tok_mapper[tok]
            elif tok not in self.okay_connected_words:
                # if the tok is something like based-on-a-true-story,
                # split on - and recurse on result.
                orig_tok = tok
                accum_tok = ""
                for cc in self.connecting_chars:
                    tok_toks = orig_tok.split(cc)
                    if len(tok_toks) > 1:
                        new_toks = self.clean_str(" ".join(tok_toks))
                        accum_tok = f"{accum_tok} {new_toks}".strip()

                tok = orig_tok if accum_tok == "" else accum_tok

            clean_toks.append(tok)
        
        clean_tag = " ".join(clean_toks)
        # print(str, " -> ", clean_tag)
        return clean_tag

    def get_words_glove_vec(self, word: str) -> tuple[bool, NDArray[float]]:

        vec = self.word_to_vec_mapping.get(word.lower(), None)
        if vec is None:
            return (False, np.zeros(self.embedded_vec_len))
        else:
            return (True, vec)

    def embed_str(self, str: str, max_num_toks: int) -> NDArray[[float, float]]:

        embedding_matrix = np.zeros((max_num_toks, self.embedded_vec_len), dtype=np.float32)
        toks = str.split()
        total_toks = len(toks)
        i = 0
        while i < min(total_toks, max_num_toks):
            embedding_matrix[i] = self.get_words_glove_vec(toks[i])[1]
            i = i + 1 

        return embedding_matrix


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
    def __init__(self, bad_data_detector=(False, lambda x: False)):
        self.reset(bad_data_detector)

    def reset(self, bad_data_detector=(False, lambda x: False)) -> None:
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
        movie_id = movies_df_row.loc[MOVIE_LENS_25M_MOVIE_ID_COL]
        #print(movie_id)
        row.append(movie_id)

        # get the title embedding, which is a numpy array of floats
        title_mat = movies_df_row.loc[MOVIE_LENS_25M_MOVIE_TITLE_COL]
        flattened_title = title_mat.flatten()
        row.extend(flattened_title)
        
        # get the genres, which is a list of ints
        row.extend(movies_df_row.loc[MOVIE_LENS_25M_MOVIE_GENRES_COL])

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

        This takes a long while to do all the movies, so it reports progress to avoid seeming stalled.
        '''

        movie_id_col = str(movie_id)
        # get user IDs and their rating for users who rated the movie with movie_id
        df = ratings_df[ratings_df[MOVIE_LENS_25M_MOVIE_ID_COL] == movie_id][[MOVIE_LENS_25M_USER_ID_COL, MOVIE_LENS_25M_RATING_COL]]
        df.columns = [MOVIE_LENS_25M_USER_ID_COL, movie_id_col]
        # make the user IDs be the row index
        df.set_index(MOVIE_LENS_25M_USER_ID_COL, inplace=True)
        # insert a row with a 0 for every user ID who did not rate movie with movie_id
        df = df.reindex(reindexer, fill_value='0')

        # save this info to the sparse data info
        col_vec = df[movie_id_col].to_numpy().astype(np.float16)
        self.add_column_from_col_vec(col_vec, quiet)

        return movie_id


def drop_zipcode_tail(zipcode: object) -> int:
    '''
    Assumes the given object is a zipcode and if it as a string has a '-' in it,
    it returns the numbers leading up to it as an int. Otherwise, it returns the
    whole thing as an int.
    '''
    str_zip = str(zipcode)
    for i in range(len(str_zip)):
        if str_zip[i] == '-':
            return np.int32(str_zip[:i])

    return np.int32(zipcode)


def genres_to_one_hot(genres: str, sep: str, all_genres: list) -> list[int]:
    '''
    Given a string of genres seperated by 'sep's, returns an encoded 1 hot vector where
    if genres contains Action - Western, in alphabetical order, there is a 1, else 0.
    Example:
    Letting all_genres =
        [ACTION,
         ADVENTURE,
         ANIMATION,
         CHILDRENS,
         COMEDY,
         CRIME,
         DOCUMENTARY,
         DRAMA,
         FANTASY,
         FILM_NOIR,
         HORROR,
         MUSICAL,
         MYSTERY,
         ROMANCE,
         SCI_FI,
         THRILLER,
         WAR,
         WESTERN]
    If sep = '|' and genres = "Action|Comedy|War|Western"
    then, the one hot encoding that represents genres is
        [1,
         0,
         0,
         0,
         1,
         0,
         0, 
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         1,
         1]
    '''
    gens = genres.split(sep)
    return [1 if g in gens else 0 for g in all_genres]

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