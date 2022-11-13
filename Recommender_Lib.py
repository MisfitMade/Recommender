import os
import gdown

# This makes PROJECT_ROOT_DIR be a complete path to the top level of the Recommender
# repo, no matter what computer you run on
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
PATH_TO_RESOURCES = os.path.join(PROJECT_ROOT_DIR, "resources")
PATH_TO_PROCESSED_DATA =  os.path.join(PATH_TO_RESOURCES, "Processed_Matrices")
PATH_TO_PROCESSED_USERS_DATA = os.path.join(PATH_TO_PROCESSED_DATA, "users.npz")
PATH_TO_PROCESSED_RATINGS_DATA = os.path.join(PATH_TO_PROCESSED_DATA, "ratings.npz")
PATH_TO_PROCESSED_MOVIES_DATA = os.path.join(PATH_TO_PROCESSED_DATA, "movies.npz")
PATH_TO_PROCESSED_DATA_SPECS = os.path.join(PATH_TO_PROCESSED_DATA, "processed_data_specs.json")
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
if os.getcwd() == PROJECT_ROOT_DIR and not procd_data_exists():
    print("Downloading necessary processed data.")
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/1dtvBO8a71j-23ih0lyyqoH34jUBaDc8t?usp=sharing",
        quiet=False)
    # make sure it worked
    if not procd_data_exists():
        raise Exception(f"Could not download processed data. Are you connected to the internet?")
SPARSE_DATA_TAG = "sparse_data_tag"
SPARSE_COORDS_TAG = "sparse_coords_tag"
USERS_MATRIX = "users_matrix"
SPEC_MOVIE_IDS = "movie_ids"
SPEC_USER_IDS = "user_ids"
SPEC_MOVIES_MATRIX_COLUMN_COUNT = "movie_mat_col_cnt"

