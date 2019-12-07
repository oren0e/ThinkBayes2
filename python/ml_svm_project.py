import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

iris = sns.load_dataset('iris')
iris.head()
iris.info()

# Create a pairplot of the data set. Which flower species seems to be the most separable?
sns.pairplot(iris, hue='species')
plt.show()  # answer: Setosa

# Create a kde plot of sepal_length versus sepal width for setosa species of flower
plt.style.use('ggplot')
sns.kdeplot(iris[iris['species']=='setosa']['sepal_width'],
            iris[iris['species']=='setosa']['sepal_length'], cmap='plasma', shade=True, shade_lowest=False
            )
plt.show()

# train test split
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Call the SVC() model from sklearn and fit the model to the training data.
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

# Now get predictions from the model and create a confusion matrix and a classification report
from sklearn.metrics import classification_report, confusion_matrix
pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# Gridsearch practice
# Import GridsearchCV from SciKit Learn
from sklearn.model_selection import GridSearchCV

# Create a GridSearchCV object and fit it to the training data.
params_to_tune = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), params_to_tune, verbose=3, refit=True)
grid.fit(X_train, y_train)
grid.best_params_

# Now take that grid model and create some predictions using the test set and create classification reports
# and confusion matrices for them. Were you able to improve?
pred = grid.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# My addition: RandomSearch practice
from sklearn.model_selection import RandomizedSearchCV

params_to_tune = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = RandomizedSearchCV(SVC(), params_to_tune, verbose=3, refit=True)
grid.fit(X_train, y_train)
grid.best_params_

pred = grid.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))