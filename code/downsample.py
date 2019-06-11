import collections
import numpy as np
import csv
from tqdm import tqdm
from nltk.corpus import stopwords 

import NBclassifier

def downsample_data(tweets, labels):
    """Downsample tweets and labels randomly to 1%, 5%, 10%, 25%, 50%, and 75% of original data.

    Writes downsampled data to files.
    
    Args:
        tweets: list of tweets
        labels: list of labels for tweets

    Returns: Nothing
    """
    
    import pdb; pdb.set_trace()
    numTweets = labels.shape[0]
    
    # shuffle all tweets and labels randomly first
    newOrder = np.random.permutation(numTweets)
    tweets = tweets[newOrder]
    labels = labels[newOrder]
    
    downsampleFracs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75])
    downsampleFilePrefixes = np.array(['01', '05', '10', '25', '50', '75'])
    for i in range(downsampleFracs.shape[0]):
        downsampleRange = int(downsampleFracs[i] * numTweets)
        
        downsampleFile = '../data/2016_train_{}.csv'
        with open(downsampleFile.format(downsampleFilePrefixes[i]), 'w+') as fp:
            for j in range(downsampleRange):
                fp.write(f'"{tweets[j]}", {labels[j]}\n')
                
                
    return

if __name__ == '__main__':
    trainTweets, trainLabels = NBclassifier.load_dataset(r'../data/2016_train.csv')
    downsample_data(trainTweets, trainLabels)