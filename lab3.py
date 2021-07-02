import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
DataFrame = pd.read_csv(path,header=None)
print(DataFrame.head(20))

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n",headers)
DataFrame.columns=headers
print(DataFrame.describe())

#Menghitung kategori yang ada di dalam kolom dataset:
drive_colume_category = DataFrame["drive-wheels"].value_counts()
#drive_colume_category.rename(columns={"drive-wheels":"value_counts"},inplace=True)
#drive_colume_category.index.name="drive-wheels"
print(drive_colume_category)


print(DataFrame["drive-wheels"].dtypes)


#cleaning data:
DataFrame.replace("?",np.nan,inplace=True)
DataFrame.dropna(subset=["price"],axis=0,inplace=True)
DataFrame["price"]=DataFrame["price"].astype("int")
print(DataFrame["price"].dtypes)


#Melihat hubungan antara "variable prediksi" dan feature
#Melihat dekripsi data menggunakan Box-Plots tujuan untuk melihat outlier data:
import seaborn as sns
sns.boxplot(x="drive-wheels",y="price",data=DataFrame)
plt.show()



#Melihat hubungan antar data menggunakan scatter grafik:
plt.scatter(DataFrame["engine-size"],DataFrame["price"])
plt.show()

#Grouping: for transform our dataset
"""
Problem: we'are interesting to know avarage price of vehicles 
and observe how they differ between different type of body-styles and drive wheels
"""
DataFrame_test = DataFrame[["drive-wheels", "body-style", "price"]]
DataFrame_group = DataFrame_test.groupby(["drive-wheels","body-style"],as_index=False).mean()


#Transform above table to pivot table:
DataFrame_pivot = DataFrame_group.pivot(index="drive-wheels",columns="body-style")
print(DataFrame_pivot)

#Menggunakan Heatmap
plt.pcolor(DataFrame_pivot,cmap='RdBu')
plt.colorbar()
plt.show()

#ANOVA mencari tahu impact yang diberikan oleh kategorical data : 
DataFrame_anova = DataFrame[["make","price"]]
group_anova = DataFrame_anova.groupby(["make"])
print(group_anova)

#Korelasi antara 2 variabel yang berbeda
#"Correlation doesn't imply causation"

sns.regplot(x="engine-size",y="price",data=DataFrame)
plt.ylim(0,)

plt.show()
sns.regplot(x="highway-mpg",y="price",data=DataFrame)
plt.ylim(0,)

plt.show()


DataFrame.dropna(subset=["peak-rpm"],axis=0,inplace=True)
DataFrame["peak-rpm"] = DataFrame["peak-rpm"].astype("int")
print(DataFrame["peak-rpm"])
sns.regplot(x="peak-rpm",y="price",data=DataFrame)
plt.ylim(0,)
plt.show()

"""
Measure the strength of the correlation between two feature using metode Person Correlation
Metode ini akan memberikan kita 2 nilai yaitu:
Correlation Coefficient dan p-value
"""
#Melihat correlasi dari howspower dan price:
from scipy.stats import pearsonr
person_coef, p_value = pearsonr(DataFrame["highway-mpg"],DataFrame["price"])
print(person_coef)
print(p_value)
print(DataFrame.corr())

print(DataFrame_group)
print(DataFrame["drive-wheels"])