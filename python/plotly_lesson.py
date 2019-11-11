import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from plotly import __version__
import cufflinks as cf
#from plotly.offline import download_plotlyjs,init_notebook_mode,iplot
import plotly.graph_objects as go

#init_notebook_mode(connected=True)
cf.go_offline()

# Data
df = pd.DataFrame(np.random.randn(100,4), columns='A B C D'.split())
df.head()

df2 = pd.DataFrame({'Category':['A','B','C'], 'Values':[32,43,50]})
df2.head()

df.iplot()
go.show()


## plotly (used in the way presented here) works on ly with Jupyter notebooks