import collections
import numpy as np
import csv
from tqdm import tqdm
from nltk.corpus import stopwords 

from NBclassifier import *

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

def downsample_analysis():
    """Evaluate how downsampled tweets and labels with semisupervised EM perform on validation and test sets.

    Writes train, val, and test accuracies to a separate file for each downsampled dataset.
    
    Args: Nothing
    
    Returns: Nothing
    """
    import pdb; pdb.set_trace()

    allTrainTweets, allTrainLabels = load_dataset(r'../data/2016_train.csv')
    valTweets, valLabels = load_dataset(r'../data/2016_val.csv')
    testTweets, testLabels = load_dataset(r'../data/2016_test.csv')
    unlabelledTweets = load_unlabelled_dataset(r'../data/unlabelled_dataset.txt')
    
    downsampleFilePrefixes = np.array(['01', '05', '10', '25', '50', '75'])
    downsampleFileName = '../data/2016_train_{}.csv'

    for prefix in downsampleFilePrefixes:
        downsampleFile = downsampleFileName.format(prefix)
        trainTweets, trainLabels = load_dataset(downsampleFile)
        
        combinedTweets = np.append(trainTweets, unlabelledTweets)
        wordDict = create_dictionary(combinedTweets, False)
        combinedMatrix = transform_text(combinedTweets, wordDict, False)
        NBModel = fit_semisupervised_naive_bayes_model(combinedMatrix, trainLabels)
        
        trainWordMatrix = transform_text(trainTweets, wordDict, False)
        trainPreds = predict_from_naive_bayes_model_3(NBModel, trainWordMatrix)
        trainAcc = np.mean(trainPreds == trainLabels)
        
        valWordMatrix = transform_text(valTweets, wordDict, False)
        valPreds = predict_from_naive_bayes_model_3(NBModel, valWordMatrix)
        valAcc = np.mean(valPreds == valLabels)
        
        testWordMatrix =  transform_text(testTweets, wordDict, False)
        testPreds = predict_from_naive_bayes_model_3(NBModel, testWordMatrix)
        testAcc = np.mean(testPreds == testLabels)
        
        resultsFile = '../data/results_{}.csv'
        with open(resultsFile.format(prefix), 'w+') as fp:
            fp.write(f'{trainAcc}, {valAcc}, {testAcc}\n')
        
if __name__ == '__main__':
    trainTweets, trainLabels = load_dataset(r'../data/2016_train.csv')
    # downsample_data(trainTweets, trainLabels)
    
    downsample_analysis()