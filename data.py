import json
import pandas as pd


def read_csv(csv_filename):
    # read data file and assign column names
    data = pd.read_csv(csv_filename)
    # remove unnecessary columns
    data = data.drop(labels=[],
                     axis='columns')

    return data