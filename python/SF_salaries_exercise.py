import pandas as pd

sal = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/04-Pandas-Exercises/Salaries.csv')

pd.set_option('expand_frame_repr', False)

sal.head()

sal.info()  # as str() in R

# What is the average basepay?
sal['BasePay'].mean()

# What is the highest amount of OvertimePay in the dataset?
sal["OvertimePay"].max()

# What is the job title of JOSEPH DRISCOLL?
# Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll)
sal.loc[sal.EmployeeName == 'JOSEPH DRISCOLL','JobTitle']
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']  # answer

# How much does JOSEPH DRISCOLL make (including benefits)?
sal.loc[sal.EmployeeName == 'JOSEPH DRISCOLL','TotalPayBenefits']
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']  # answer


# What is the name of highest paid person (including benefits)?
sal.loc[sal['TotalPayBenefits'].idxmax()]
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']  # answer


# What is the name of lowest paid person (including benefits)?
# Do you notice something strange about how much he or she is paid?
sal.loc[sal['TotalPayBenefits'].idxmin()]
sal.iloc[sal['TotalPayBenefits'].argmin()]  # answer

# What was the average (mean) BasePay of all employees per year? (2011-2014)?
sal.groupby('Year')['BasePay'].mean()

# How many unique job titles are there?
sal['JobTitle'].nunique()
len(sal['JobTitle'].unique())

# What are the top 5 most common jobs?
sal.groupby('JobTitle')['JobTitle'].count().nlargest(5)
sal['JobTitle'].value_counts().head(5)  # answer

# How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)
sum(sal[sal.Year == 2013].groupby(['JobTitle'])['JobTitle'].count() == 1)
sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)


# How many people have the word Chief in their job title? (This is pretty tricky)
sum(sal['JobTitle'].str.contains(".*Chief.*", regex = True))


def cheif_string(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False

sum(sal['JobTitle'].apply(lambda x: cheif_string(x)))

# Bonus: Is there a correlation between length of the Job Title string and Salary?
sal['JobTitle_len'] = sal['JobTitle'].apply(len)
sal[['JobTitle_len', 'TotalPayBenefits']].corr()


