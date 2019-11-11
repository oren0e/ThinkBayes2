import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                  '07-Pandas-Built-in-Data-Viz/df1',index_col=0)
df1.head()

df2 = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                  '07-Pandas-Built-in-Data-Viz/df2')
df2.head()

df1['A'].hist(bins=30)
plt.show()

df1['A'].plot(kind='hist')  # one way to call plots out of a dataframe in pandas
plt.show()

df1['A'].plot.hist()  # second way to plot
plt.show()

df2.plot.area(alpha=0.4)
plt.show()

df2.plot.bar(stacked=True)
plt.show()

df1['A'].plot.hist(bins=50)
plt.show()

df1.plot.line(y='B', figsize=(12,3),linewidth=1)
plt.show()

df1.plot.scatter(x='A',y='B', c='C',cmap='coolwarm')
plt.show()

df1.plot.scatter(x='A',y='B', s=df1['C']*100)
plt.show()

df2.plot.box()
plt.show()

df = pd.DataFrame(np.random.randn(1000,2),columns=['a','b'])
df.head()
df.plot.hexbin(x='a',y='b',gridsize=25, cmap='coolwarm')
plt.show()

df2['a'].plot.kde()
plt.show()

df2['a'].plot.density()
plt.show()

df2.plot.density()
plt.show()