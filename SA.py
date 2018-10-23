import csv
import re

TRAIN_DATA="./GOLD/Subtask_A/twitter-2013train-A.txt"
TEST_DATA="./GOLD/Subtask_A/twitter-2013test-A.txt"
DEV_DATA="./GOLD/Subtask_A/twitter-2013dev-A.txt"

trainData = []
testData = []
devData = []

with open(TRAIN_DATA) as file:
    trainData = file.readlines()
with open(TEST_DATA) as file:
    testData = file.readlines()
with open(DEV_DATA) as file:
    devData = file.readlines()

def removePattern(tweet, pattern):
    r = re.findall(pattern, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    return tweet 

def preprocess(data):
    for line in data:
        tId, tSent, tweet = line.split("\t")
        print(tweet)
        tweet = removePattern(tweet, "@[\w]*")
        print(tweet)
        tweet = tweet.replace("[^a-zA-Z#]", " ")
        print(tweet)

preprocess(trainData)
