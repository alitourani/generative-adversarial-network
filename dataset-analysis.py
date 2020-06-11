# Show the Dataset contents in a Table

# Importing the required libraries
import pandas as pd 
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
sns.set(color_codes=True)

# Loading the CSV file into a pandas dataframe.
dataFeatures = pd.read_csv("/content/drive/My Drive/CarsInCsvFormat.csv")

# Removing irrelevant columns of the dataset
dataFeatures = dataFeatures.drop(["DriveTrain", "Cylinders", "MPG_City", "MPG_Highway", "Weight", "Wheelbase", "Length"], axis=1)

# Describe the data
# dataFeatures.describe()

# Identify the type of data (such as null, float, etc)
# dataFeatures.info()

# Finding duplicate data
# dataFeatures = dataFeatures.drop_duplicates(subset="MSRP", keep="first")
# dataFeatures.count()

# Finding the null values
# print(dataFeatures.isnull().sum())

# Showing a subset of data
dataFeatures.head(20)