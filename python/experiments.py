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

# Dataframes - Part 2
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
df['States'] = newind
df

df.set_index('States')
df

# Dataframes - Part 3
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

df = pd.DataFrame(randn(6,2),index=hier_index,columns=['A','B'])
df
df.loc['G1'].loc[1]
df.index.names
df.index.names = ['Groups','Num']
df

df.loc['G2'].loc[2]['B']
df
df.loc['G1'].iloc[2]['A']

df
df.xs('G1')
df.xs(1,level='Num')

# Missing Data
