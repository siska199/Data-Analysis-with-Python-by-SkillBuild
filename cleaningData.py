import pandas as pd
import numpy as np

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
missing_value = ["?", np.nan,"N/a","N/A"]
df_siska = pd.read_csv(path,header=None,na_values=missing_value)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n",headers)
df_siska.columns=headers
print(df_siska.dtypes)
print(df_siska.isnull().sum())
print(df_siska)

