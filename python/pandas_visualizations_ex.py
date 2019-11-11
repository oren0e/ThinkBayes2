import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                  '07-Pandas-Built-in-Data-Viz/df3')

df3.head()

df3.plot.scatter(x='a',y='b',c='red',figsize=[10,3])
plt.show()

df3['a'].plot.hist(bins=30,alpha=0.5)
plt.style.use('ggplot')
plt.show()

df3[['a','b']].plot.box()
plt.show()

df3['d'].plot.kde(linewidth=3,linestyle='--')
plt.show()

plt.figure()
df3.iloc[0:31,:].plot.area()
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()