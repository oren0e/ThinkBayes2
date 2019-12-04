import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/'
                 'KNN_Project_Data')
df.head()

# Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column
sns.pairplot(df, hue='TARGET CLASS')
plt.show()

# standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
scaled_features

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

# Train and test split
from sklearn.model_selection import train_test_split
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# knn modeling
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# predictions and evaluations
pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))

# Choosing K value
# Let's go ahead and use the elbow method to pick a good K Value!
# Create a for loop that trains various KNN models with different k values,
# then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

error_rate.index(np.min(error_rate))  # lowest for K = 30

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color="blue", markersize=10, linestyle="--", markerfacecolor="red", marker="o")
plt.title('Error Rate vs K value')
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()
#  Based on the plot K=17 is the best choice
# Retrain your model with the best K value (up to you to decide what you want)
# and re-do the classification report and the confusion matrix

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)

# predictions and evaluations
pred = knn.predict(X_test)
print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))

