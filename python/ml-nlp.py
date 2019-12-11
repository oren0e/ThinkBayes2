import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#nltk.download_shell()
messages = [line.rstrip() for line in open('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/'
                                           'Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/'
                                           'smsspamcollection/SMSSpamCollection',encoding='utf-8')]

print(len(messages))
messages[0]

# print first 10 messages and number them using enumerate
for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')

# first column is label and second column message itself
messages = pd.read_csv('/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/'
                                           'Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/'
                                           'smsspamcollection/SMSSpamCollection',encoding='utf-8', sep='\t',
                       names=['label','message'])
messages.head()

# EDA #
messages.describe()

# start with a higher level view of the data just to get intuition for what separates spam messages from ham messages
messages.groupby('label').describe()

# think about the feature engineering
messages['length'] = messages['message'].apply(len)
messages.head()

# plot the messages and the length of it
plt.style.use('ggplot')
messages['length'].plot.hist(bins=50,edgecolor="black")
plt.show()

# lets explore why there are very long text messages
messages['length'].describe()
messages[messages['length'] == 910]['message'].iloc[0]

# lets see if message length is a distinctive feature between spam and ham
messages.hist(column='length', by='label', bins=60, figsize=(12,4), edgecolor='black')
plt.show()  # looks like spam messages are longer than ham on average

# Part 2 #
# in this section we will convert the sequences of characters to sequences of numbers
# we are going to write a function that will split the message into its words and return a list
# also we will remove stopwords
import string

# we want to remove punctuation first, lets see how this is going to work:
mess = "Sample message! Notice: it has punctuation."
nopunc = [c for c in mess if c not in string.punctuation]
nopunc
# removing stopwords
from nltk.corpus import stopwords
stopwords.words('english')
nopunc = ''.join(nopunc)  # a way for joining elements in a list together
nopunc.split()
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess
# Now lets put these together into a function

def text_process(mess):
    """
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# tokenize the messages (i.e. perform the procedures we just did)
messages['message'].head(5).apply(text_process)  # example
# There are a lot more ways to process the text, for example stemming and lemmatization
# now we will focus on vectorization of the words to represent them with numbers

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)

# which words appear more than once in mess4 (from looking at print(bow4)
bow_transformer.get_feature_names()[4068]
bow_transformer.get_feature_names()[9554]

# Part 3 #


