
# coding: utf-8

# In[38]:


import csv
import re
import codecs


# In[39]:


TRAIN_DATA = "./GOLD/Subtask_A/twitter-2013train-A.txt"
TEST_DATA = "./GOLD/Subtask_A/twitter-2013test-A.txt"
DEV_DATA = "./GOLD/Subtask_A/twitter-2013dev-A.txt"


# In[44]:


trainData = []
testData = []
devData = []

def readData(path):
    data = []
    with open(path) as file:
        data = file.read()
        data = codecs.decode(data, 'unicode_escape')
        data = data.split('\n')[:-1]
    return data

trainData = readData(TRAIN_DATA)
testData = readData(TEST_DATA)
devData = readData(DEV_DATA)


# In[45]:


def removePattern(tweet, pattern):
    r = re.findall(pattern, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    return tweet 

def preprocess(data):
    cleanData = []
    for line in data:
        tId, tSent, tweet = line.split("\t") # Splitting by tabspace
        tweet = removePattern(tweet, "@[\w]*") # Removing @user tags
        tweet = tweet.replace("[^a-zA-Z#]", " ") # Removing punctuation and special characters
        tweet = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',"<URL>", tweet)
        cleanData.append([tId, tSent, tweet])
    return cleanData


# In[46]:


trainData = preprocess(trainData)
testData = preprocess(testData)
devData = preprocess(devData)

