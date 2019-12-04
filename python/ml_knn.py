import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/'
                 'Classified Data',index_col=0)
df.head()

# standardize variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # create an instance like a ML model
scaler.fit(df.drop('TARGET CLASS',axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))  # performs the standardization
scaled_features

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])  # without the last one
df_feat.head()

from sklearn.model_selection import train_test_split
X = df_feat  # or scaled_features
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# use KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)  # K=1
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))

# use the elbow method to choose a correct K value
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color="blue", linestyle="--", marker = "o", markerfacecolor="red", markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

knn = KNeighborsClassifier(n_neighbors=17)  # K=1
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test,pred))