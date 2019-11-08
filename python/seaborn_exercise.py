import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

titanic = sns.load_dataset('titanic')
titanic.head()

sns.jointplot(x='fare',y='age',data=titanic,kind='scatter')
plt.show()

sns.distplot(titanic['fare'],bins=30,kde=False,color='red')
plt.show()

sns.boxenplot(x='class',y='age',data=titanic)
plt.show()

sns.swarmplot(x='class',y='age',data=titanic)
plt.show()

sns.countplot(x='sex', data=titanic)
plt.show()

sns.heatmap(titanic.corr(),cmap='coolwarm')
plt.title('titanic.corr()')
plt.show()

g = sns.FacetGrid(titanic,col='sex')
g.map(plt.hist,'age')
#g.map(sns.distplot,'age',kde=False,color='blue')
plt.show()