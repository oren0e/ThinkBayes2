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
