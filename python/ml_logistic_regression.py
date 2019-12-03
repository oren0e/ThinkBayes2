import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

train = pd.read_csv("/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv")

train.head()
train.info()
train.describe()

# missing values information
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

sns.set_style('whitegrid')

# who survived and who didn't
sns.countplot(data=train, x='Survived')
plt.show()

sns.countplot(data=train, hue='Sex', x='Survived', palette='RdBu_r')
plt.show()

sns.countplot(data=train, hue='Pclass', x='Survived')
plt.show()

sns.distplot(train['Age'].dropna(), kde=False, bins=30)
plt.show()

# same thing
train['Age'].plot.hist(bins=35)
plt.show()

sns.countplot(x='SibSp',data=train)
plt.show()

train['Fare'].hist(bins=40, figsize=(10,4))
plt.show()

## Part 2 ##
# Imputation of missing values
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age', data=train)
plt.show()

# simple imputation of Age by mean age per class
train['Age'].groupby(train['Pclass']).mean().round(2)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
           return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# too many missings in Cabin column so drop it
train.drop('Cabin',axis=1,inplace=True)
train.info()

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
train.dropna(inplace=True)

# Converting categorical variables to dummies
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()

# Concatenate these into our main data frame
train = pd.concat([train,sex,embark],axis=1)  # axis=1 because we add them as new columns
train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
train.drop('PassengerId',axis=1,inplace=True)

# Part 3 #
X = train.drop('Survived', axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

# evaluation #
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

