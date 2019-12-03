import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

ad_data = pd.read_csv("/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv")

ad_data.head()
ad_data.info()
ad_data.describe()

# Create a histogram of the Age #
sns.distplot(ad_data['Age'], bins=35, kde=False)
plt.show()

# Create a jointplot showing Area Income versus Age
sns.jointplot(x='Age', y='Area Income', data=ad_data)
plt.show()

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind="kde",color='red')
plt.show()

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage
sns.jointplot(y='Daily Internet Usage', x='Daily Time Spent on Site', data=ad_data, color='green')
plt.show()

# Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(ad_data, hue='Clicked on Ad')
plt.show()

# check for missing values
sns.heatmap(ad_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Feature Engineering
city_dummies_df = pd.get_dummies(ad_data['City'],prefix="city",drop_first=True)
country_dummies_df = pd.get_dummies(ad_data['Country'],prefix="country",drop_first=True)
ad_data1 = pd.concat([ad_data,city_dummies_df,country_dummies_df],axis=1)  # p > n so we wont use it

# at which time the measurement was taken (morning afternoon or evening)
ad_data['hour'] = pd.to_datetime(ad_data['Timestamp']).apply(lambda x: x.hour)

def time_of_day(num):
    if (num >= 6) & (num < 12):
        return "morning"
    elif (num >= 12) & (num < 18):
        return "noon_afternoon"
    else:
        return "evening_night"

ad_data['time_of_day'] = ad_data['hour'].apply(time_of_day)
ad_data['time_of_day'].value_counts()
time_of_day_df = pd.get_dummies(ad_data['time_of_day'],drop_first=True)
ad_data1 = pd.concat([ad_data,time_of_day_df],axis=1)
ad_data1.head()

ad_data1.drop(['Ad Topic Line','City','Country','Timestamp','hour','time_of_day'],axis=1,inplace=True)

# Training the model
from sklearn.model_selection import train_test_split
X = ad_data1.drop('Clicked on Ad',axis=1)
y = ad_data1['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Predictions and Evaluation (threshold is 0.5!)
pred = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,pred))