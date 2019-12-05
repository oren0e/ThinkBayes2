import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                 '15-Decision-Trees-and-Random-Forests/kyphosis.csv')

df.head()
df.info()

# EDA
sns.pairplot(df, hue='Kyphosis')
plt.show()

# ML
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# single decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

pred = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# random forrest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))

df['Kyphosis'].value_counts(normalize=True)