import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.keys()

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df_feat.head()

# train test split
from sklearn.model_selection import train_test_split
X = df_feat
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# train SVM
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print("\n")
print(classification_report(y_test, pred))

# use GridSearch
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)
# best parameters
grid.best_params_
grid.best_score_

# rerun the predictions just as we would normally with model object
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print("\n")
print(classification_report(y_test, grid_predictions))
