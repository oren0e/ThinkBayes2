import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()

from sklearn.preprocessing import StandardScaler
# scale our data so that each feature has a 1 unit variance
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# PCA workflow is simialr to the scaler - we "fit" pca object and then use the transform method with one of the param being how many
# components we want

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape  # reduced 30 dimensions to 2

# visualize
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1])
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show()

# based on only the first 2 principle components we can get a pretty good separation in terms of the target values
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'])
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show()

# get the ingredients of the components
pca.components_  # each row is a principal component and each column relates back to the features

df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
df_comp.head()  # the 2 principal components

plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')
plt.show()  # the higher the number (the hotter the color) the more correlated the relevant component to the specific feature in the columns

