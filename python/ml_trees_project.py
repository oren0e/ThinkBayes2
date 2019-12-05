import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

loans = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                 '15-Decision-Trees-and-Random-Forests/loan_data.csv')

loans.info()
loans.head()
loans.describe()

# EDA
# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.
# Note: This is pretty tricky, feel free to reference the solutions.
# You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()
plt.style.use('ggplot')
plt.figure(figsize=(8,6))
loans['fico'][loans['credit.policy'] == 1].hist(alpha=0.5,color="blue",bins=30,edgecolor="black", label='Credit Policy = 1')
loans['fico'][loans['credit.policy'] == 0].hist(alpha=0.5,color="red",bins=30,edgecolor="black", label='Credit Policy = 0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# Create a similar figure, except this time select by the not.fully.paid column.
plt.style.use('ggplot')
plt.figure(figsize=(8,6))
loans['fico'][loans['not.fully.paid'] == 1].hist(alpha=0.5,color="blue",bins=30,edgecolor="black",label='Not Fully Paid = 1')
loans['fico'][loans['not.fully.paid'] == 0].hist(alpha=0.5,color="red",bins=30,edgecolor="black",label='Not Fully Paid = 0')
plt.legend()  # ?
plt.show()

# Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
plt.figure(figsize=(10,6))
sns.countplot(x='purpose', hue='not.fully.paid',data=loans)
plt.show()

# Let's see the trend between FICO score and interest rate. Recreate the following jointplot
sns.jointplot(x='fico', y='int.rate',data=loans, color='purple')
plt.show()

# Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy.
# Check the documentation for lmplot() if you can't figure out how to separate it into columns.
sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid')
plt.show()

# Setting up the data
loans.info()
# Notice that the purpose column as categorical
# That means we need to transform them using dummy variables so sklearn will be able to understand them.
# Let's do this in one clean step using pd.get_dummies.
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()

# train test split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# training a decision tree
from sklearn.tree import DecisionTreeClassifier
# Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

pred = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=600)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print(confusion_matrix(y_test, pred_rf))
print('\n')
print(classification_report(y_test, pred_rf))
