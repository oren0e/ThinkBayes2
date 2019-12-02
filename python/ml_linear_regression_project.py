import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr', False)

import matplotlib.pyplot as plt
import seaborn as sns



customers = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/'
                 'Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/Ecommerce_Customers.csv')

customers.head()
customers.info()
customers.describe()

# Exploratory Data Analysis (EDA)
# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)
plt.show()

# Do the same but with the Time on App column instead
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)
plt.show()

# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
sns.jointplot(x="Time on App", y="Length of Membership", data=customers, kind="hex")
plt.show()

# Let's explore these types of relationships across the entire data set.
# Use pairplot to recreate the plot below.(Don't worry about the the colors)
sns.pairplot(customers)
plt.show()

# Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
# Answer: Length of Membership

# Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership
sns.lmplot(x="Length of Membership",y="Yearly Amount Spent", data=customers)
plt.show()

# Training and Testing Data #
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# Set a variable X equal to the numerical features of the customers and a variable y equal
# to the "Yearly Amount Spent" column.
X = list(customers.select_dtypes(exclude='object').columns)
X.remove('Yearly Amount Spent')
X = customers[X]
y = customers['Yearly Amount Spent']

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets.
# Set test_size=0.3 and random_state=101
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train the model
# Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Create an instance of a LinearRegression() model named lm
lm = LinearRegression()

# Train/fit lm on the training data
lm.fit(X_train,y_train)

# Print out the coefficients of the model
lm.coef_

# Predict Test data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# Use lm.predict() to predict off the X_test set of the data.
pred = lm.predict(X_test)

# Create a scatterplot of the real test values versus the predicted values.
res_df = pd.DataFrame(np.transpose([pred, y_test]), columns=['prediction','real'])
sns.scatterplot(x='real', y='prediction', data=res_df)
plt.show()

# Evaluating the model #
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
# Refer to the lecture or to Wikipedia for the formulas
from sklearn import metrics
MAE = metrics.mean_absolute_error(res_df['real'], res_df['prediction'])
MSE = metrics.mean_squared_error(res_df['real'], res_df['prediction'])
RMSE = np.sqrt(MSE)
print(f"\nMAE: {MAE}\nMSE: {MSE}\nRMSE: {RMSE}")

# R^2
metrics.explained_variance_score(res_df['real'], res_df['prediction'])

# Residuals #
# Plot a histogram of the residuals and make sure it looks normally distributed.
# Use either seaborn distplot, or just plt.hist()
sns.distplot((res_df['real'] - res_df['prediction']))
plt.show()

# Coefficients table #
cdf = pd.DataFrame(np.transpose([X.columns,lm.coef_]),columns=['Variable Name','Coefficient'])