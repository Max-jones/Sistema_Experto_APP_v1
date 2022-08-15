# load_data.py

import pandas as pd
#  
import config

from load_local_data import load_data

# DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))[:-9] #without notebooks
csv_file = config.TRAINING_FILE
df = load_data(csv_file)
# print(df.head())


# df.head()