import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

df = pd.read_csv("./python/Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/911.csv")
df.info()
df.head()

# What are the top 5 zipcodes for 911 calls?
df['zip'].value_counts().head(5)

# What are the top 5 townships (twp) for 911 calls?
df['twp'].value_counts().head(5)

# Take a look at the 'title' column, how many unique title codes are there?
df['title'].head()
df['title'].nunique()

'''
In the titles column there are "Reasons/Departments" specified before the title code. 
These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called 
"Reason" that contains this string value
For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.
'''
df['Reason'] = df['title'].str.split(':').apply(lambda x: x[0])

# What is the most common Reason for a 911 call based off of this new column?
df['Reason'].value_counts().head(3)

# Now use seaborn to create a countplot of 911 calls by Reason.
sns.countplot(x='Reason',data=df)
plt.show()

# Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?
df['timeStamp'].dtypes

#  You should have seen that these timestamps are still strings.
#  Use pd.to_datetime to convert the column from strings to DateTime objects.
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

#You can now grab specific attributes from a Datetime object by calling them. For example:**

df['timeStamp'].iloc[0].hour


# Now that the timestamp column are actually DateTime objects,
# use .apply() to create 3 new columns called Hour, Month, and Day of Week.
# You will create these columns based off of the timeStamp column
df['Hour'] = df['timeStamp'].dt.hour
df['Month'] = df['timeStamp'].dt.month
df['Day_of_week'] = df['timeStamp'].dt.dayofweek

# Notice how the Day of Week is an integer 0-6. Use the .map()
# with this dictionary to map the actual string names to the day of the week:
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day_of_week'] = df['Day_of_week'].map(dmap)

#  Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.
sns.countplot(x='Day_of_week', hue='Reason', data=df)
plt.show()

# Now do the same for Month
sns.countplot(x='Month', hue='Reason', data=df)
plt.show()

'''
You should have noticed it was missing some Months, let's see if we can maybe fill in this information
by plotting the information in another way, possibly a simple line plot that fills in the missing months,
in order to do this, we'll need to do some work with pandas...
'''

# Now create a gropuby object called byMonth, where you group the DataFrame by the month
# column and use the count() method for aggregation. Use the head() method on this returned DataFrame.
byMonth = df.groupby('Month').count()
byMonth.head()

# Now create a simple plot off of the dataframe indicating the count of calls per month.
byMonth.plot(y='lat', figsize=(12,3),linewidth=2)
plt.show()

# Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month.
# Keep in mind you may need to reset the index to a column.