import pandas as pd
pd.set_option('expand_frame_repr', False)  # To view all the variables in the console
df = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/'
                 'Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/iris.csv')
df.head()

# for TF the target needs to be an integer and the variable names cant have spaces or special characters
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
df.head()
df['target'] = df['target'].apply(int)
df.head()

y = df['target']
X = df.drop('target',axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

import tensorflow as tf
# 1. Create Feature Columns list for the tf estimator
X.columns
feat_cols = []
for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
feat_cols

# 2. Create Input Function (actually will be 2 input functions - one for training and one for evaluation)
"""
It is not a good idea to feed the network with all the training cases because it can crash, so it's better to feed it in batches.
A good indicator to alter the number of batches is by TF giving an "error" - which usually presents itself as 
empty predictions or null values. 1 epoch means that you gone through all of your training data 1 time.
Shuffle for data that its target is sorted (actually train_test_split shuffles by default)
"""
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=5, shuffle=True)

# 3. Create the Estimator (the classifier)
classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10], n_classes=3, feature_columns=feat_cols)  # will have 3 hidden layers
# with 10 neurons in the first, 20 on the
# second and 10 on the third

# 4. Train the Estimator
classifier.train(input_fn=input_func, steps=50)  # behind the scenes it creates the graph that we did in the manual lecture

# 5. Evaluate
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)  # no need to pass in batches
# because we want to evaluate only, so we do it all in one large batch
pred = list(classifier.predict(input_fn=pred_fn))  # this is a Generator so we want to cast that into a list
pred
final_preds = []
for p in pred:
    final_preds.append(p['class_ids'][0])
final_preds

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, final_preds))
print('\n')
print(classification_report(y_test, final_preds))




