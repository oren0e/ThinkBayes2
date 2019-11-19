from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

tickers = {'boa': 'BAC',
           'cg': 'C',
           'gs': 'GS',
           'jpm': 'JPM',
           'ms': 'MS',
           'wf': 'WFC'}
dfs = {}
for key, ticker in tickers.items():
    dfs[key] = data.get_data_tiingo(ticker, start='2006-01-01', end='2016-01-01',
                                   api_key='e779f083cb1660c3befb3e9850e05f13f78ce2c0')

tickers_lst = list(tickers.values())

# Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks.
# Set the keys argument equal to the tickers list. Also pay attention to what axis you concatenate on.
bank_stocks = pd.concat([dfs[key] for key in dfs.keys()], keys=[ticker for ticker in tickers_lst],axis=1)
bank_stocks = bank_stocks.xs(list(itertools.product(tickers_lst,var)),axis=1)

# What is the max Close price for each bank's stock throughout the time period?
bank_stocks1 = pd.read_pickle('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/' \
                              'Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/all_banks')
bank_stocks1.xs(list(itertools.product(tickers_lst,['Close'])),axis=1).max()
# We can use pandas pct_change() method on the Close column to create a column representing this return value.
# Create a for loop that goes and for each Bank Stock Ticker creates this returns
# column and set's it as a column in the returns DataFrame
returns = pd.DataFrame()
returns = bank_stocks1.xs(list(itertools.product(tickers_lst,['Close'])),axis=1).pct_change()

# Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?
sns.pairplot(returns)
sns.set
plt.show()

# Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns.
# You should notice that 4 of the banks share the same day for the worst drop,
# did anything significant happen that day?

returns.xs(list(itertools.product(tickers_lst,['Close'])),axis=1).idxmax()  # best
returns.xs(list(itertools.product(tickers_lst,['Close'])),axis=1).idxmin()  # worst

# Take a look at the standard deviation of the returns, which stock would you classify
# as the riskiest over the entire time period?
# Which would you classify as the riskiest for the year 2015?
returns.xs(list(itertools.product(tickers_lst,['Close'])),axis=1).std()  # the entire time period
returns_with_date = returns.xs(list(itertools.product(tickers_lst,['Close'])),axis=1).reset_index()
returns_with_date[returns_with_date['Date'].dt.year == 2015].xs(list(itertools.product(tickers_lst,['Close'])),axis=1).std()

# Create a distplot using seaborn of the 2015 returns for Morgan Stanley
g = sns.distplot(returns_with_date[returns_with_date['Date'].dt.year == 2015].xs('MS',axis=1),
             bins=50,color='green')
g.set_xlabel('MS Return')
plt.show()

# Create a distplot using seaborn of the 2008 returns for CitiGroup
g = sns.distplot(returns_with_date[returns_with_date['Date'].dt.year == 2008].xs('C',axis=1),
             bins=100,color='red')
g.set_xlabel('C Return')
plt.show()

# Create a line plot showing Close price for each bank for the entire index of time.
# (Hint: Try using a for loop, or use .xs to get a cross section of the data.)*
sns.set_style('whitegrid')
bank_stocks1_with_date = bank_stocks1.stack().reset_index()[bank_stocks1.stack().reset_index()['Stock Info'] == 'Close']
bank_stocks1_with_date.plot
plt.show()