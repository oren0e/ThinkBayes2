from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import itertools

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

