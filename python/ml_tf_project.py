import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console
df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/'
                 'python/Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/bank_note_data.csv')
df.head()

# EDA #
# Create a Countplot of the Classes (Authentic 1 vs Fake 0)
sns.countplot(x='Class', data=df)
plt.show()

sns.pairplot(df, hue='Class')
plt.show()

# Data Preparation #
# When using Neural Network and Deep Learning based systems, it is usually a good idea to Standardize your data,
# this step isn't actually necessary for our particular data set, but let's run through it for practice!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Class', axis=1))
scaled_feats = scaler.transform(df.drop('Class',axis=1))
scaled_feats = pd.DataFrame(scaled_feats, columns=df.columns.drop('Class'))

# train test split
from sklearn.model_selection import train_test_split
X = scaled_feats
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Tensorflow #
# Create a list of feature column objects using tf.feature.numeric_column() as we did in the lecture
feat_cols = []
for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
feat_cols

# Create an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10]
# hidden unit layer structure:
classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10], n_classes=2, feature_columns=feat_cols)

# Now create a tf.estimator.pandas_input_fn that takes in your X_train, y_train, batch_size and set shuffle=True.
# You can play around with the batch_size parameter if you want, but let's start by setting it to 20 since our data isn't very big.
train_fun = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, shuffle=True)

# Now train classifier to the input function. Use steps=500. You can play around with these values if you want!
classifier.train(input_fn=train_fun, steps=500)

# Model Evaluation #
# Create another pandas_input_fn that takes in the X_test data for x. Remember this one won't need any y_test info since we will
# be using this for the network to create its own predictions. Set shuffle=False since we don't need to shuffle for predictions.
pred_fun = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)

#  Use the predict method from the classifier model to create predictions from X_test
predictions = list(classifier.predict(input_fn=pred_fun))

# Now create a classification report and a Confusion Matrix. Does anything stand out to you?
predictions
final_preds = [pred['class_ids'][0] for pred in predictions]
final_preds

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,final_preds))
print('\n')
print(classification_report(y_test,final_preds))

# Use SciKit Learn to Create a Random Forest Classifier and compare the confusion matrix and classification report to the DNN model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(confusion_matrix(y_test,rf_pred))
print('\n')
print(classification_report(y_test,rf_pred))



