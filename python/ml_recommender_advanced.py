import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)  # To view all the variables in the console
columns_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                 '19-Recommender-Systems/u.data', sep='\t', names=columns_names)

movie_titles = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                 '19-Recommender-Systems/Movie_Id_Titles')
movie_titles.head()

# merge the data frames
df = pd.merge(df,movie_titles,on='item_id')
df.head()

n_users = df.user_id.nunique()
n_items = df.item_id.nunique()
print('Num of Users: '+ str(n_users))
print('Num of Movies: '+ str(n_items))

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

# Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

# You can use the pairwise_distances function from sklearn to calculate the cosine similarity.
# Note, the output will range from 0 to 1 since the ratings are all positive.
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        # Simply put, the newaxis expression is used to increase the dimension of the existing array by one more dimension, when used once.
        ratings_diff = (ratings - mean_user_rating[:,np.newaxis])  # makes mean_user_rating a column vector
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')






