import csv
import numpy as np
import re
import string

"""Preprocess the 2016-2018 Climate Change Stance Tweets dataset from a CSV file
Remove punctuation, converts tweets to lowervase, removes @mentions and 'RT' from
start of retweets. Removes all emojis and hperlinks. Splits dataset into 70% train,
10% dev and 20% test.
"""

def load_2016_dataset(csv_path):
    tweets = []
    labels = []

    with open(csv_path, 'r', newline='', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        # skip header line
        next(reader)

        for label, tweet in reader:
            if label == '2': #skip factual news about climate change category
                continue
            tweets.append(tweet)
            if label == '0': # the tweet neither supports nor refutes the belief of man-made climate change
                labels.append(0)
            elif label == '1': # the tweet supports the belief of man-made climate change
                labels.append(1) 
            elif label == '-1': # the tweet refutes the belief in man-made climate change
                labels.append(-1) 
            else:
                raise Exception('Unknown labels in dataset! Please check that labels are -1, 0, 1, 2.')
    return np.array(tweets), np.array(labels)

def load_unlabelled_tweets(txt_path):
    tweets = []

    with open(txt_path, 'r',encoding='utf8') as txt_file:
        # skip header line
        for tweet in txt_file:
            tweets.append(tweet)
    return np.array(tweets)

def preprocess_tweets(tweets, labels = None):
    cleaned_tweets = []
    matching_labels = []
    for i in range(len(tweets)):
        tweet = tweets[i].lower()
        tweet = tweet.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        tweet = tweet.replace('\n', ' ').replace('\r', '') # remove new lines
        if tweet[:2] == 'rt': continue # ignore retweets      
        tweet = re.sub(r'http[^ ]* ?', ' ', tweet) # remove links
        tweet = tweet.encode('ascii', 'ignore').decode('ascii') # can do this since we're only interested in English tweets without emojis
        if len(tweet) <5: continue
        cleaned_tweets.append(tweet)
        if labels is not None:
            matching_labels.append(labels[i])
    return np.array(cleaned_tweets), np.array(matching_labels)

def split_dataset(tweets, labels, testFrac = 0.1, valFrac = 0.2):
    """Split dataset into training, validation and test sets which are written to separate files

    Args:
        tweets: Numpy array of tweets in dataset
        labels: Numpy array of labels for each tweet in dataset
        testFrac: fraction of tweets and labels to assign to test set
        valFrac: fraction of tweets and labels to assign to validation set

    Creates three files, train.csv, val.csv, and test.csv containing the three sets of data
    """
    numTweets = len(tweets)
    print(numTweets)
    groupAssignments = np.zeros(numTweets)
    testValInd = np.random.choice(numTweets, int((testFrac+valFrac)*numTweets), replace=False)
    
    testInd = np.random.choice(testValInd, int(testFrac*numTweets), replace=False)
    valInd = np.setdiff1d(testValInd,testInd)
    groupAssignments[testInd] = 1
    groupAssignments[valInd] = 2
    trainMask = groupAssignments == 0
    
    traintweets = tweets[trainMask]
    trainLabels = labels[trainMask]
    valtweets = tweets[valInd]
    valLabels = labels[valInd]
    testtweets = tweets[testInd]
    testLabels = labels[testInd]
    
    with open('train_new.csv', 'w+') as fp:
        for i in range(len(traintweets)):
            fp.write(f'"{traintweets[i]}", {trainLabels[i]}\n')

    with open('val_new.csv', 'w+') as fp:
        for i in range(len(valtweets)):
            fp.write(f'"{valtweets[i]}", {valLabels[i]}\n')
    
    with open('test_new.csv', 'w+') as fp:
        for i in range(len(testtweets)):
            fp.write(f'"{testtweets[i]}", {testLabels[i]}\n')
  
def main():
    csv_path = r'data\climatechangestance.csv'
    txt_path = r'data\unlabelled_raw_tweets.txt'

    tweets = load_unlabelled_tweets(txt_path)
    tweets, _ = preprocess_tweets(tweets)
    tweets = list(set(tweets))
    with open('unlabelled.txt', 'w+') as fp:
        for i in range(len(tweets)):
            fp.write(f'{tweets[i]}\n')
    
    tweets, labels = load_2016_dataset(csv_path)
    tweets, labels = preprocess_tweets(tweets, labels)
    split_dataset(tweets, labels, testFrac = 0.1, valFrac = 0.2)
    
if __name__ == '__main__':
    main()
