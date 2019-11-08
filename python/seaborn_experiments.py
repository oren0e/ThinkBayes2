import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
tips.head()

#### Distributional plots ####
sns.distplot(tips['total_bill'])
plt.show()
sns.distplot(tips['total_bill'],kde=False,bins=30)
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='regression')
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')
plt.show()

sns.pairplot(tips, hue='sex',palette='coolwarm')
plt.show()

sns.rugplot(tips['total_bill'])
plt.show()

#### Categorical plots ####
sns.barplot(x='sex',y='total_bill',data=tips)
plt.show()

import numpy as np
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)
plt.show()

sns.countplot(x='sex',data=tips)  # counting the number of instances of each category
plt.show()

sns.boxenplot(x='day',y='total_bill',data=tips,hue='smoker')
plt.show()

sns.violinplot(x='day',y='total_bill',data=tips, hue='sex',split=True)
plt.show()

sns.stripplot(x='day',y='total_bill',data=tips,jitter=True,hue='sex',split=True)  # split based on the hue argument
plt.show()

sns.swarmplot(x='day',y='total_bill',data=tips)  # shows the distribution as well (not good for large datasets)
plt.show()

# not that explainable to management but for exploratory data analysis
# for management use the first plots like boxplots, bar plots etc.
sns.violinplot(x='day',y='total_bill',data=tips)
sns.swarmplot(x='day',y='total_bill',data=tips,color='black')  # shows the distribution as well (not good for large datasets)
plt.show()

sns.factorplot(x='day',y='total_bill',data=tips, kind='bar')
plt.show()

sns.factorplot(x='day',y='total_bill',data=tips, kind='violin')
plt.show()

#### Matrix plots ####
flights = sns.load_dataset('flights')
flights.head()

tc = tips.corr()

sns.heatmap(tc,annot=True,cmap='coolwarm')
plt.show()

fp = flights.pivot_table(index='month', columns='year',values='passengers')
sns.heatmap(fp,linecolor='white',linewidths=1,cmap='coolwarm')
plt.show()

sns.clustermap(fp,cmap='coolwarm',standard_scale=1)
plt.show()

#### Grid plots ####
iris = sns.load_dataset('iris')
iris.head()
iris['species'].unique()

sns.pairplot(iris)
plt.show()

g = sns.PairGrid(iris)
g.map(plt.scatter)
plt.show()

g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()

g = sns.FacetGrid(tips,col='time',row='smoker')
g.map(sns.distplot,'total_bill')
plt.show()

g = sns.FacetGrid(tips,col='time',row='smoker')
g.map(plt.scatter,'total_bill','tip')
plt.show()

#### Regression plots ####
sns.lmplot(x='total_bill',y='tip',data=tips, hue='sex')
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips, hue='sex',markers=['o','v'],
           scatter_kws={'s':100})
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,col='sex',row='time')  # essentially does the Grid from the previous section
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,col='day',row='time',hue='sex')
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex', aspect=0.6,size=8)
plt.show()

#### Style and Color ####
sns.set_style('darkgrid')
sns.countplot(x='sex',data=tips)
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='sex',data=tips)
plt.show()

sns.set_style('ticks')
sns.countplot(x='sex',data=tips)
sns.despine()
plt.show()

sns.set_style('ticks')
sns.countplot(x='sex',data=tips)
sns.despine(left=True, bottom=True)
plt.show()

plt.figure(figsize=(12,3))
sns.countplot(x='sex',data=tips)
plt.show()

sns.set_context('poster',font_scale=2)
sns.countplot(x='sex',data=tips)
plt.show()

sns.set_context('notebook')
sns.countplot(x='sex',data=tips)
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='seismic')
plt.show()