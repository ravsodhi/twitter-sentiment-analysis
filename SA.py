
# coding: utf-8

# In[1]:


import csv
import re
import codecs

import numpy as np

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score


# In[2]:


TRAIN_DATA = "./GOLD/Subtask_A/twitter-2013train-A.txt"
TEST_DATA = "./GOLD/Subtask_A/twitter-2013test-A.txt"
DEV_DATA = "./GOLD/Subtask_A/twitter-2013dev-A.txt"


# In[3]:


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


# In[4]:


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
#         tweet = tokenize(tweet)
        cleanData.append([tId, tSent, tweet])
        
    return cleanData

def tokenize(tweet):
    return TweetTokenizer().tokenize(tweet)


# In[5]:


en_stopwords = set(stopwords.words("english")) 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)


# In[6]:


trainData = preprocess(trainData)
testData = preprocess(testData)
devData = preprocess(devData)


# In[9]:


trainTweets = [x[2] for x in trainData]
X_train = np.array(trainTweets)
trainSents = [x[1] for x in trainData]
y_train = []
for x in trainSents:
    if x == "negative":
        y_train.append(-1)
    elif x == "neutral":
        y_train.append(0)
    elif x == "positive":
        y_train.append(1)
        
testTweets = [x[2] for x in testData]
X_test = np.array(testTweets)
testSents = [x[1] for x in testData]
y_test = []
for x in testSents:
    if x == "negative":
        y_test.append(-1)
    elif x == "neutral":
        y_test.append(0)
    elif x == "positive":
        y_test.append(1)
# X = vectorizer.fit_transform(trainTweets)
# X


# In[ ]:


# kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# np.random.seed(1)
# pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced"))
# grid_svm = GridSearchCV(pipeline_svm,
#                     param_grid = {'svc__C': [0.01, 0.1, 1]}, 
#                     cv = kfolds,
#                     verbose=1,   
#                     n_jobs=-1) 
# grid_svm.fit(X_train, y_train)
# grid_svm.score(X_test, y_test)


# In[ ]:


# grid_svm.best_params_


# In[ ]:


# grid_svm.best_score_


# In[ ]:


# def report_results(model, X, y):
#     pred_proba = model.predict_proba(X)[:, 1]
#     pred = model.predict(X)        

# #     auc = roc_auc_score(y, pred_proba)
#     acc = accuracy_score(y, pred)
# #     f1 = f1_score(y, pred)
# #     prec = precision_score(y, pred)
# #     rec = recall_score(y, pred)
# #     result = {'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
#     result  = {'acc':acc}
#     return result


# In[ ]:


# report_results(grid_svm.best_estimator_, X_test, y_test)


# In[10]:


X = np.append(X_train, X_test)
X = vectorizer.fit_transform(X)
n = X_train.shape[0]
X_train = X[:n]
X_test = X[n:]


# In[11]:


nsv = SVC(probability=True, kernel='linear',class_weight="balanced", C = 0.1)
nsv.fit(X_train, y_train)
nsv.score(X_test, y_test)

