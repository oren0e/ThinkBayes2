import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/'
                 '17-K-Means-Clustering/College_Data', index_col=0)

df.head()
df.info()
df.describe()

sns.scatterplot(x='Room.Board', y='Grad.Rate', hue='Private',data=df)
plt.show()

sns.scatterplot(x='Outstate', y='F.Undergrad', hue='Private',data=df)
plt.show()

plt.style.use('ggplot')
df[df['Private']=='Yes']['Outstate'].plot(kind='hist',color="blue", edgecolor="black", alpha=0.3, bins=25)
df[df['Private']=='No']['Outstate'].plot(kind='hist', color="red", edgecolor="black", alpha=0.3, bins=25)
plt.xlabel('Outstate')
plt.show()

# Facetgrid way
g = sns.FacetGrid(df, hue='Private', palette='coolwarm',size=6, aspect=2)
g = g.map(plt.hist,'Outstate',bins=20, alpha=0.7,edgecolor='black')
plt.show()

plt.style.use('ggplot')
df[df['Private']=='Yes']['Grad.Rate'].plot(kind='hist',color="blue", edgecolor="black", alpha=0.3, bins=25)
df[df['Private']=='No']['Grad.Rate'].plot(kind='hist', color="red", edgecolor="black", alpha=0.3, bins=25)
plt.xlabel('Grad.Rate')
plt.show()

# Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?
df.head()
df[df['Grad.Rate'] > 100]

# Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error)
# when doing this operation, so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.
df.loc[df['Grad.Rate']>100, 'Grad.Rate'] = 100
# the warning way
df['Grad.Rate']['Cazanovia College'] = 100


# K-Means clustering
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model.fit(df.drop('Private',axis=1))

# What are the cluster center vectors?
model.cluster_centers_
model.labels_

# Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.
df['Cluster'] = np.where(df['Private'] == 'Yes', 1, 0)

# apply method way
def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Cluster'],model.labels_))
print("\n")
print(classification_report(df['Cluster'],model.labels_))