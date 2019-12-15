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

# Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class.
# (Your features and target/labels)
X = yelp_class['text']
y = yelp_class['stars']

# Import CountVectorizer and create a CountVectorizer object
from sklearn.feature_extraction.text import CountVectorizer
yelp_text_count = CountVectorizer()
# Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X
X = yelp_text_count.fit_transform(raw_documents=X)

# Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training the model
# Import MultinomialNB and create an instance of the estimator and call is nb
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# Now fit nb using the training data.
nb.fit(X_train, y_train)
pred = nb.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test,pred))

# Let's see what happens if we try to include TF-IDF to this process using a pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

# Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()
model_pipe = Pipeline([('count_vectorizer', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('naive_bayes', MultinomialNB())])
# Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already,
# meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version.
# What we need is just the text
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Now fit the pipeline to the training data. Remember you can't use the same training data as last time because
# that data has already been vectorized. We need to pass in just the text and labels
model_pipe.fit(X_train, y_train)
pred = model_pipe.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test,pred))
