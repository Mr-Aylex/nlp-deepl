import ast
import pandas as pd
import numpy as np

# TODO: Create a function that reads in the data from the raw data folder and returns the optimal dataframes

def make_dataset(filename):
    df = pd.read_csv(filename, dtype={"is_comic": int, "is_name": object, "video_name": str})
    # is_name is str, convert is_name to list of bool
    df["is_name"] = df["is_name"].apply(ast.literal_eval).apply(lambda x: np.array(x, dtype=np.float32))
    return df
