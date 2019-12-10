import numpy as np
import pandas as pd

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

columns_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                 '19-Recommender-Systems/u.data', sep='\t', names=columns_names)
df.head()

movie_titles = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                 '19-Recommender-Systems/Movie_Id_Titles')
movie_titles.head()

# merge the data frames
df = pd.merge(df,movie_titles,on='item_id')
df.head()

# best rated movies
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
# movies with the most ratings
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

# ratings data frame with average rating and number of ratings
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
# rating (avg rating) depends on how many people have rated it
ratings['num_of_ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

# visualization
ratings['num_of_ratings'].hist(bins=70)
plt.show()
ratings['rating'].hist(bins=70)
plt.show()
# the relatioinship between the average rating and the number of ratings
sns.jointplot(x='rating', y='num_of_ratings', data=ratings, alpha=0.5)
plt.show()

# create a matrix with user_ids on one axis and the movie titles on another axis
# (each cell will consist of the rating the user gave to the movie)
df.head()
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
moviemat.head()
ratings.sort_values('num_of_ratings',ascending=False).head(10)
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_usr_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_usr_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.info()
corr_starwars.head()

corr_starwars.sort_values('Correlation', ascending=False).head(10)  # does not make sense because its probably users that rated one movie
# and also rated star wars as 5.0, let's fix this by considering only above certain number of reviews
corr_starwars = corr_starwars.join(ratings['num_of_ratings'])  # we did this by the index, that why we used join and not merge
corr_starwars.head()

corr_starwars[corr_starwars['num_of_ratings'] > 100].sort_values('Correlation', ascending=False).head(10)

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num_of_ratings'])
corr_liarliar.head()
corr_liarliar[corr_liarliar['num_of_ratings'] > 100].sort_values('Correlation', ascending=False).head(10)