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

