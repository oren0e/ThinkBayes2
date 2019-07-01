# ------- DataFrames with Pandas ---------

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
d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}
df = pd.DataFrame(d)
df

df.dropna()
df.dropna(axis=1)
df.dropna(thresh=2)

df
df.fillna(value='FILL VALUE')
df['A'].fillna(value=df['A'].mean())

# Group by
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
        'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
        'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
df
byComp = df.groupby('Company')
byComp.mean()
byComp.sum()
byComp.std()

byComp.sum().loc['FB']
df.groupby('Company').sum().loc['FB']

df.groupby('Company').count()
df.groupby('Company').max()

df.groupby('Company').describe()
df.groupby('Company').describe().transpose()
df.groupby('Company').describe().transpose()['FB']
df.groupby('Company').describe().transpose()['FB'][0]

# Merging, joining, and concatenating
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])
#%%
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])
#%%
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

df1
df2
df3

pd.concat([df1,df2,df3])  # like rbind
pd.concat([df1,df2,df3],axis=1)  # like cbind

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

left
right
pd.merge(left,right,how='inner',on='key')

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
left
right
pd.merge(left,right,on=['key1','key2'])
pd.merge(left,right,how='left',on=['key1','key2'])

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])

left
right
left.join(right,how='outer')


# Operations





