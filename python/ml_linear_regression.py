import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr', False)

import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/'
                 'Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv')

df.head()
df.info()
df.describe()

# Names of columns
df.columns

# plots to check out the data
sns.pairplot(df)
plt.show()

sns.distplot(df['Price'])
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

# split to X and Y
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

# train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# import model
from sklearn.linear_model import LinearRegression

# instantiate a model instance
lm = LinearRegression()

# train model
lm.fit(X_train, y_train)  # This already takes affect on the object itself so no need to assign to a variable

print(lm.intercept_)
lm.coef_

# create a dataframe with coefficients
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf

## Part 2 ##
predictions = lm.predict(X_test)
predictions
y_test

# plot predictions vs actual
plt.scatter(y_test,predictions)
plt.show()

# dist plot of residuals
sns.distplot((y_test-predictions))
plt.show()

# 3 main evaluation metrics for regression problems (MAE, MSE, RMSE)
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))
