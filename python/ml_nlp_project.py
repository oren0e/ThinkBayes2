import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

yelp = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/'
                   'Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv')
yelp.head()
yelp.info()
yelp.describe()

# Create a new column called "text length" which is the number of words in the text column.
yelp['text_length'] = yelp['text'].apply(len)
yelp.head()

# EDA #
# Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings.
# Reference the seaborn documentation for hints on this
g = sns.FacetGrid(yelp,col='stars')
g = g.map(plt.hist,'text_length',bins=20,edgecolor='black')
plt.show()

# Create a boxplot of text length for each star category.
sns.boxplot(x='stars', y='text_length', data=yelp)
plt.show()

# Create a countplot of the number of occurrences for each type of star rating.
sns.countplot(x='stars', data=yelp)
plt.show()

# Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation
yelp.head()
yelp.info()

stars_df = yelp.select_dtypes('number').groupby('stars').mean()

# correlation dataframe
# Use the corr() method on that groupby dataframe to produce this dataframe:
stars_df.corr()
# Then use seaborn to create a heatmap based off that .corr() dataframe
sns.heatmap(data=stars_df.corr(), cmap='coolwarm', annot=True)
plt.show()

# NLP classification task #
# Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
yelp_class.head()

