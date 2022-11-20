import os
import gdown

import pandas as pd
import numpy as np

from scipy import sparse
from numpy.typing import NDArray
from numpy import dot
from numpy.linalg import norm

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

# This makes PROJECT_ROOT_DIR be a complete path to the top level of the Recommender
# repo, no matter what computer you run on
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
PATH_TO_RESOURCES = os.path.join(PROJECT_ROOT_DIR, "resources")
PATH_TO_PROCESSED_DATA =  os.path.join(PATH_TO_RESOURCES, "Processed_Data")
PATH_TO_PROCESSED_USERS_DATA = os.path.join(PATH_TO_PROCESSED_DATA, "users.npz")
PATH_TO_PROCESSED_RATINGS_DATA = os.path.join(PATH_TO_PROCESSED_DATA, "ratings.npz")
PATH_TO_PROCESSED_MOVIES_DATA = os.path.join(PATH_TO_PROCESSED_DATA, "movies.npz")
PATH_TO_PROCESSED_DATA_SPECS = os.path.join(PATH_TO_PROCESSED_DATA, "processed_data_specs.json")
PATH_TO_USER_EMBEDDING_SPECS = os.path.join(PATH_TO_RESOURCES, "user_embedding_specs.json")
PATH_TO_COSINE_SIMS_PARTITION_SPECS = os.path.join(PATH_TO_RESOURCES, "partition_specs.json")
PATH_TO_COSINE_SIM_PARTITIONS = os.path.join(PATH_TO_RESOURCES, "cosine_sim_partitions")
PATH_TO_USER_KMEANS_MODEL = os.path.join(PATH_TO_RESOURCES, "kmeans.pickle")
SPARSE_DATA_TAG = "sparse_data_tag"
SPARSE_COORDS_TAG = "sparse_coords_tag"
COSINE_SIMS_TAG = "cosine_sims"
USERS_MATRIX = "users_matrix"
SPEC_MOVIE_IDS = "movie_ids"
SPEC_USER_IDS = "user_ids"
SPEC_MOVIES_MATRIX_COLUMN_COUNT = "movie_mat_col_cnt"
SPEC_NUM_TOKS_EMBEDDED = "toks_embedded"
SPEC_EMBEDDING_LEN = "embedding_len"
SPEC_UNIQUE_VOCAB = "unique_user_vocab"
SPEC_OUTPUT_DIM = "output_dim"
SPEC_ZIPCODE_LEN = "zipcode_len"
SPEC_MALE_FEMALE = "gender_encodes"
SPEC_NUM_USERS = "num_users"
SPEC_PARTITIONS = "num_partitions"
SPEC_PARTITION_PATHS = "partition_paths"
SPEC_ROWS_PER_PARTITION = "num_rows_per_partition"
SPEC_EXTRA_ROWS_IN_LAST_PARTITION = "num_extra_rows_in_last_partition"


# Movie data strs
PATH_TO_MOVIE_LENS = os.path.join(PATH_TO_RESOURCES, "MovieLens")
PATH_TO_MOVIE_LENS_1M = os.path.join(PATH_TO_MOVIE_LENS, "1Mill")
PATH_TO_MOVIE_LENS_25M = os.path.join(PATH_TO_MOVIE_LENS, "25Mill")
PATH_TO_MOVIE_LENS_1M_MOVIES = os.path.join(PATH_TO_MOVIE_LENS_1M, "movies.dat")
PATH_TO_MOVIE_LENS_25M_MOVIES = os.path.join(PATH_TO_MOVIE_LENS_25M, "movies.csv")
PATH_TO_MOVIE_LENS_1M_RATINGS = os.path.join(PATH_TO_MOVIE_LENS_1M, "ratings.dat")
PATH_TO_MOVIE_LENS_25M_RATINGS = os.path.join(PATH_TO_MOVIE_LENS_25M, "ratings.csv")
PATH_TO_MOVIE_LENS_1M_USERS = os.path.join(PATH_TO_MOVIE_LENS_1M, "users.dat")
PATH_TO_MOVIE_LENS_25M_TAGS = os.path.join(PATH_TO_MOVIE_LENS_25M, "tags.csv")
PATH_TO_GLOVE = os.path.join(PATH_TO_RESOURCES, "Glove_twitter_vers")

# Users strings
MOVIE_LENS_25M_USER_ID_COL = "userId"
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
MOVIE_LENS_1M_DELIM = "::"
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

# Ratings strings
MOVIE_LENS_25M_USER_ID_COL = "userId"
MOVIE_LENS_25M_RATING_COL = "rating"
MOVIE_LENS_25M_TIMESTAMP_COL = "timestamp"
MOVIE_LENS_25M_RATINGS_COLS = [
    MOVIE_LENS_25M_USER_ID_COL,
    MOVIE_LENS_25M_MOVIE_ID_COL,
    MOVIE_LENS_25M_RATING_COL,
    MOVIE_LENS_25M_TIMESTAMP_COL]

def read_raw_data(
    short_circuit_for_movie_titles=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    There is users data, ratings data and movies data from two different data sets.
    This function reads all that in, then cominbes the users w/ users, ratings w/
    ratings and movies w/ movies to make 3 parsed dfs
    '''
    verify_raw_data_exists()

    ratings_1M_df = pd.read_table(
        PATH_TO_MOVIE_LENS_1M_RATINGS,
        sep=MOVIE_LENS_1M_DELIM, # sep="::"
        header=None, # tell the read that the data has no col headers
        engine="python") # explicit use of python engine to use > 1 length sep ("::") without a warning
    ratings_1M_df.columns = MOVIE_LENS_25M_RATINGS_COLS

    ratings_25M_df = pd.read_csv(PATH_TO_MOVIE_LENS_25M_RATINGS)

    movies_df, ratings_1M_df = read_raw_movies(ratings_1M_df)
    if short_circuit_for_movie_titles:
        return movies_df[[MOVIE_LENS_25M_MOVIE_ID_COL, MOVIE_LENS_25M_MOVIE_TITLE_COL]], None, None

    users_df, ratings_df = read_raw_users(ratings_1M_df, ratings_25M_df)

    return movies_df, users_df, ratings_df


def encode_age(age: int) -> int:

    if age < 18: return 1
    elif age < 25: return 18
    elif age < 35: return 25
    elif age < 45: return 35
    elif age < 50: return 45
    else: return 56


def encode_genders(gender: str, male_female_encoder) -> int:

    if gender == 'M' or 'm': return int(male_female_encoder[0])
    elif gender == 'F' or 'f': return int(male_female_encoder[1])
    else: return int(gender)


def encode_occupation(occ: str) -> int:

    if type(occ) is str:
        occ = occ.lower()
        if "academic" in occ or "educator" in occ or "teacher" in occ or "professor" in occ: return 1
        elif "artist" in occ or "painter" in occ or "sculpter" in occ: return 2
        elif "clerical" in occ or "admin" in occ: return 3
        elif "student" in occ:
            if "college" in occ or "grad" in occ: return 4
            elif "elementary" in occ or "middle school" in occ or "high school" in occ or "K-12" in occ: return 10 
        elif "customer service" in occ: return 5
        elif "doctor" in occ or "health care" in occ or "nurse" in occ: return 6
        elif "executive" in occ or "manage" in occ: return 7
        elif "farmer" in occ or "rancher" in occ: return 8
        elif "homemaker" in occ: return 9
        elif "lawyer" in occ or "attorney" in occ: return 11
        elif "programmer" in occ or "software" in occ: return 12
        elif "retired" in occ: return 13
        elif "sales" in occ or "marketing" in occ: return 14
        elif "scientist" in occ or "chemist" in occ or "physicist" in occ: return 15
        elif "self-employed" in occ or "self employed" in occ: return 16
        elif "technician" in occ or "engineer" in occ: return 17
        elif "tradesman" in occ or "craftsman" in occ or "plumber" in occ or "electrician" in occ: return 18
        elif "unemployed" in occ: return 19
        elif "writer" in occ: return 20
        else: return 0
    else:
        # else, assume it is an int and already encoded
        return occ


def read_raw_movies(
    ratings_1M_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Reads, parses and combines the two movie data set's worth of data into a single df
    and makes the ratings_1M_df's movie IDs match with the combined movie df's movie ids
    '''

    '''
    Read each .dat file into a DF and give them col headers to match the .csv 25 mil data sets
    '''
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

    return movies_df, ratings_1M_df


def read_raw_users(
    ratings_1M_df: pd.DataFrame,
    ratings_25M_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Reads the users data and uses the the ratings data to make a larger users data
    df, then combines the two ratings into a single ratings df
    '''
    users_df = pd.read_table(
        PATH_TO_MOVIE_LENS_1M_USERS,
        sep=MOVIE_LENS_1M_DELIM,
        header=None,
        engine="python")
    users_df.columns = MOVIE_LENS_1M_USERS_COLS

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

    users_df = pd.concat(
        [pd.DataFrame({
            MOVIE_LENS_25M_USER_ID_COL : ratings_25M_df[MOVIE_LENS_25M_USER_ID_COL].unique()}),
            users_df],
        ignore_index=True).fillna(0)

    '''
    Goal is to have the number 0 be "not specified" or "no rating" and
    need to encode F vs M numerically, so using 1 vs -1, instead of 1 vs 0.
    Also, there are some zip codes that are like 12345-6789 and I want them to just be able
    to be straight numbers, so truncate the -6789 from the zip codes that are like that.
    '''
    users_df[MOVIE_LENS_1M_ZIPCODE_COL] = users_df[MOVIE_LENS_1M_ZIPCODE_COL].map(drop_zipcode_tail)
    users_df[MOVIE_LENS_1M_GENDER_COL] = users_df[MOVIE_LENS_1M_GENDER_COL].map(
        lambda g: 1 if g == "F" else -1)

    ratings_df = pd.concat([ratings_25M_df, ratings_1M_df], ignore_index=True)

    return users_df, ratings_df


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


def get_index_of_or_of_similair(id: int, list, step: int) -> int:
    '''
    Gets the index of a int element in the list. If the element is not in the given
    list, it tries getting the int element of id+step.
    This makes the assumption that encodings close together in value are close together
    in alikeness, which may not be true.
    '''
    try:
        return list.index(id)
    except:
        stepped = id+step
        print("No direct user embedding found in vocab for ", id, "\nTrying: ", stepped)
        return get_index_of_or_of_similair(stepped, list, step)


def embed_users(
    users_df,
    zipcode_len,
    output_dim,
    input_len,
    unique_vocab=[]):
    
    '''
    In case of zipcodes that are not 5 digit zips,
    dont want some zips to have weight over others after already truncating the zips
    that were like 12345-6789, so go through and do truncation to those that are like
    123456789
    '''
    users_df[3] = users_df[3].map(
        lambda z: int(str(z)[:zipcode_len]) if len(str(z)) > zipcode_len else z)

    '''
    Also, for the embedding layer, want to bin the values sorta so that the vocab size is accurate.
    '''
    if len(unique_vocab) == 0:
        unique_v = []
        unique_v.extend(users_df[0].unique().astype(int))
        unique_v.extend(users_df[1].unique().astype(int))
        unique_v.extend(users_df[2].unique().astype(int))
        unique_v.extend(users_df[3].unique().astype(int))
        unique_v.sort()
        [unique_vocab.append(x) for x in unique_v if x not in unique_vocab]

    
    users_df[0] = users_df[0].map(lambda x: get_index_of_or_of_similair(int(x), unique_vocab, -1))
    users_df[1] = users_df[1].map(lambda x: get_index_of_or_of_similair(int(x), unique_vocab, -1))
    users_df[2] = users_df[2].map(lambda x: get_index_of_or_of_similair(int(x), unique_vocab, -1))
    users_df[3] = users_df[3].map(lambda x: get_index_of_or_of_similair(int(x), unique_vocab, -1))

    vocab_size = len(unique_vocab)

    '''
    Now can see that all the numbers in users_df are between 0 and total number of unique numbers
    '''
    print(users_df.to_numpy())

    embedder = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=output_dim,
        input_length=input_len,
        mask_zero=True)
    model = tf.keras.Sequential([embedder])
    model.compile('rmsprop', 'mse')

    users_embedding = model.predict(users_df.to_numpy())

    # Notice this gives a 4x168 embedding for each user, thus we have a Nx4x168
    print("Shape after embedding: ", users_embedding.shape)
    users_embedding_Nxk = []
    for i in range(users_df.shape[0]):
        users_embedding_Nxk.append(users_embedding[i].flatten())
    users_embedding_Nxk = np.array(users_embedding_Nxk)
    print("Shape after flatten: ", users_embedding_Nxk.shape)

    return (unique_vocab, users_embedding_Nxk)


def get_cos_sims(
    idx_of_common_vect: int,
    all_row_vects: NDArray[NDArray[np.float32]]) -> NDArray[np.float32]:
    '''
    Computes the cosine similairity of one vector compared to all other vector.
    Uses numpy vector math as much as possible to speed it up.
    '''
    common_vect = all_row_vects[idx_of_common_vect]
    denoms = norm(common_vect) * norm(all_row_vects, axis=1)
    numers = common_vect.dot(all_row_vects.T)
    
    return (numers / denoms)


def write_partition(
    partition_start: int,
    partition_end: int,
    row_vects) -> str:

    print("Starting partition: ", partition_start, "-", partition_end)
    sims = []
    for i in range(partition_start, partition_end):
        sims.append(get_cos_sims(i, row_vects))

    sim_mat_path_p = os.path.join(
        PATH_TO_COSINE_SIM_PARTITIONS,
        f"cosine_sims{partition_start}_thru_{partition_end-1}.npz")
    np.savez(sim_mat_path_p, cosine_sims=np.array(sims, dtype=np.float32))
    
    return sim_mat_path_p


def movie_title_to_mat_idx(title: str, movies_df: pd.DataFrame) -> list[int]:
    '''
    Given a movie title, returns the index that that movie is in the
    movies_matrix or ratings_matrix
    '''
    return movies_df.index[movies_df[MOVIE_LENS_25M_MOVIE_TITLE_COL].map(
        lambda t: str(t).lower().startswith(title.lower()))].tolist()


def get_col_idxs_from_float16_sparse(
    col_idxs,
    sparse_matrix) -> NDArray[np.float16]:
    '''
    sparse matrices do not allow for slicing when the type of the matrix is
    np.float16!! So, make our own.
    '''
    cols = []
    for i in col_idxs:
        col_i = sparse_matrix[:,i].astype(np.float32).toarray().astype(np.float16)
        cols.append(col_i)

    return np.column_stack(cols)


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


class RatingsMatrixManager:
    def __init__(
        self,
        path_to_Sparse_mat: str,
        movie_title_series: pd.Series,
        num_rows: int):

        ratings_data = np.load(path_to_Sparse_mat)
        sparse_ratings_coords = ratings_data[SPARSE_COORDS_TAG]
        self.ratings_mat = sparse.csr_matrix(
            (ratings_data[SPARSE_DATA_TAG], (sparse_ratings_coords[0], sparse_ratings_coords[1])),
            shape=(num_rows, movie_title_series.shape[0]))

        self.movies_series = movie_title_series
        self.liked_movie_to_likes_mapping = {}

    def add_to_likes_mapping(self, movie_idx_to_rating_pair) -> None:

        self.liked_movie_to_likes_mapping[movie_idx_to_rating_pair[0]] = (
            movie_idx_to_rating_pair[1] + (
                self.liked_movie_to_likes_mapping[movie_idx_to_rating_pair[0]] if movie_idx_to_rating_pair[0] in self.liked_movie_to_likes_mapping else 0))

    def find_movies_liked_by_alike_users(
        self,
        idxs_of_alike_users,
        number_of_alike_movies,
        thresh=4.0):

        # reset the liked movie mapping
        self.liked_movie_to_likes_mapping = {}
        # each idx = row = user
        alike_users = self.ratings_mat[idxs_of_alike_users]
        # get the non zero element's info
        rowCoords_colCoords_data = sparse.find(alike_users)
        is_highs = rowCoords_colCoords_data[2] >= thresh
        idxs_of_high_ratings = np.nonzero(is_highs)[0]

        movie_idxs_of_high_ratings = rowCoords_colCoords_data[1][idxs_of_high_ratings]
        high_ratings = rowCoords_colCoords_data[2][is_highs]

        for i in range(len(movie_idxs_of_high_ratings)):
            self.add_to_likes_mapping((movie_idxs_of_high_ratings[i], high_ratings[i]))

        # Now get the movies that have the highest like summed ratings
        highest_movie_idxs = np.zeros(number_of_alike_movies)
        highest_movie_rating_sums = np.zeros(number_of_alike_movies)
        for key, value in self.liked_movie_to_likes_mapping.items():
            if highest_movie_rating_sums.min() < value:
                replace_idx = highest_movie_rating_sums.argmin()
                highest_movie_rating_sums[replace_idx] = value
                highest_movie_idxs[replace_idx] = key

        return self.movies_series.iloc[highest_movie_idxs].to_list()
    

    def find_movies_liked_by_users_that_like_movie(
        self,
        idx_of_movie_liked,
        number_of_liked_movies,
        thresh=4.0):

        idxs_of_users_that_like_liked_movie = np.nonzero(self.ratings_mat[:,idx_of_movie_liked] >= thresh)[0]
        return self.find_movies_liked_by_alike_users(
            idxs_of_users_that_like_liked_movie,
            number_of_liked_movies)


class PartitionedCosineSimMatrix:
    def __init__(
        self,
        partition_paths,
        num_partitions,
        rows_per_partition,
        extra_rows_in_last_partition):

        self.partition_paths = partition_paths
        self.extra_rows_in_last_partition = extra_rows_in_last_partition
        self.partition_idx_strts = np.empty(num_partitions, dtype=int)
        for p in range(num_partitions):
            self.partition_idx_strts[p] = p*rows_per_partition

    def get_similar_movie_idxs(
        self,
        idx_of_common_movie: int,
        thresh: float):

        partition_idxs = np.nonzero(self.partition_idx_strts<=idx_of_common_movie)[0]
        if len(partition_idxs) == 0:
            raise Exception("The given row vector index is out of bounds.")

        # else
        partition_idx = partition_idxs[-1]
        current_partition = np.load(
            self.partition_paths[partition_idx],
            allow_pickle=True)[COSINE_SIMS_TAG]
        idx_of_matrix_vector_in_partition = idx_of_common_movie - self.partition_idx_strts[partition_idx]

        cos_sims = current_partition[idx_of_matrix_vector_in_partition]
        idx_filter = np.nonzero(cos_sims >= thresh)[0]

        return idx_filter


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




def verify_procd_data_exists() -> None:
    '''
    if the processed data does not exist and we need it because we are running in a certain directory
    download it from google drive. Otherwise, this lib is being imported by the DataAugmetation_Lib.py,
    in which case, the processed data does not exist cuz it is still being made?
    '''
    procd_data_exists = lambda: (os.path.exists(PATH_TO_PROCESSED_DATA)
        and os.path.exists(PATH_TO_PROCESSED_USERS_DATA)
        and os.path.exists(PATH_TO_PROCESSED_RATINGS_DATA)
        and os.path.exists(PATH_TO_PROCESSED_MOVIES_DATA)
        and os.path.exists(PATH_TO_PROCESSED_DATA_SPECS))
    if not procd_data_exists():
        # make it so the folder is downloaded to the resources folder
        os.chdir(PATH_TO_RESOURCES)
        print("Downloading necessary processed data.")
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1dtvBO8a71j-23ih0lyyqoH34jUBaDc8t?usp=sharing",
            quiet=False)
        # make sure it worked
        if not procd_data_exists():
            raise Exception(f"Could not download processed data. Are you connected to the internet?")
        
        os.chdir(PROJECT_ROOT_DIR)


def verify_raw_data_exists() -> None:
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
        # make it so the folder is downloaded to the resources folder
        cwd = os.getcwd()
        os.chdir(PATH_TO_RESOURCES)
        print("Downloading necessary raw data.")
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1KNYIV7wmIHAFeoc41VIiX7yt7-FlgZfF?usp=sharing",
            quiet=False)
        # make sure it worked
        if not raw_data_exists():
            raise Exception(f"Could not download raw data. Are you connected to the internet?")

        # change back to the original cwd
        os.chdir(cwd)


def verify_glove_data_exists() -> None:
    # if the path to the glove components do not exist, download them from google drive
    glove_data_exists = lambda: (os.path.exists(PATH_TO_GLOVE)
        and os.path.exists(os.path.join(PATH_TO_GLOVE, "25d.txt"))
        and os.path.exists(os.path.join(PATH_TO_GLOVE, "50d.txt"))
        and os.path.exists(os.path.join(PATH_TO_GLOVE, "100d.txt"))
        and os.path.exists(os.path.join(PATH_TO_GLOVE, "200d.txt")))
    if not glove_data_exists():
        # make it so the folder is downloaded to the resources folder
        cwd = os.getcwd()
        os.chdir(PATH_TO_RESOURCES)
        print("Downloading necessary Glove components.")
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1iddQ-LoQ-ynEK-OjpDJJiCvBtBfWuYql?usp=sharing",
            quiet=False)
        # make sure it worked
        if not glove_data_exists():
            raise Exception(f"Could not download glove components. Are you connected to the internet?")

        # change back to the original cwd
        os.chdir(cwd)


def verify_cosine_sims_mat_exists(partition_paths) -> None:

    check_partitions_exist = lambda: all(list(map(os.path.exists, partition_paths)))
    if not check_partitions_exist():
    # make it so it is downloaded to the resources folder
        cwd = os.getcwd()
        os.chdir(PATH_TO_RESOURCES)
        print("Downloading necessary raw data.")
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1jAWolVxxzb5QyjkMzVUPAmtrKm0vyhzl?usp=sharing",
            quiet=False)

        # make sure it worked
        if not check_partitions_exist():
            raise Exception(f"Could not download matrix partitions. Are you connected to the internet?")

        # change back to the original cwd
        os.chdir(cwd)
