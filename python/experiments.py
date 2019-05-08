import pandas as pd
import numpy as np
from numpy.random import randn

# Series
labels = ['a', 'b', 'c']
my_data = [10, 20, 30]
arr = np.array(my_data)
d = {'a': 10, 'b': 20, 'c': 30}

pd.Series(data = my_data, index=labels)
pd.Series(arr,labels)
pd.Series(d)

pd.Series(data=labels)
pd.Series(data=[sum,print,len])

ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
ser2 = pd.Series([1,2,5,4],['USA','Germany','Italy','Japan'])
ser3 = pd.Series(data=labels)

# Dataframes
np.random.seed(101)
df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])

df['new'] = df['W'] + df['Y']
df.drop('new',axis=1,inplace=True)
df.drop('E')
df.shape

df.loc['C']
df.iloc[2]

df
df.loc[['A','B'],['W','Y']]

booldf = df > 0
df[booldf]

resultdf = df[df['W'] > 0]
df[df['Z'] < 0]
resultdf['X']
df[df['W'] > 0]['X']

df[(df['W']>0) & (df['Y']>1)]

df.reset_index()

newind = 'CA NY WY OR CO'.split()
