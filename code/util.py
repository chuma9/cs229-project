import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

def plot_confusion_matrix(y_actual, y_pred, title='None', cmap=plt.cm.gray_r):
    """ Plot a confudion matrix for the true and predicted labels
    """
    df_confusion = pd.crosstab(y_actual, y_pred)
    plt.rcParams.update({'font.size': 22})
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    #plt.show()
    plt.savefig(title+'.png')

def load_unlabelled_dataset(txt_path):
    """ Load the unlabelled dataset
    """
    tweets = []

    with open(txt_path, 'r', encoding='utf8') as txt_file:
        for tweet in txt_file:
            tweets.append(tweet[1:-1]) # remove quotes
    return np.array(tweets)


def load_dataset(csv_path):
    """Load a CSV file containing a dataset and return a numpy array of 
    tweets and another of corresponding labels

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        tweets: A list of string values containing the text of each tweet.
        labels: The binary labels (-1, 0 or 1) for each tweet.
        A 0 indicates the tweet neither supports nor refutes the belief of man-made climate change. 
        A 1 indicates the tweet supports the belief of man-made climate change. 
        A -1 indicates the tweet refutes the belief in man-made climate change.
    """
    
    tweets = []
    labels = []

    with open(csv_path, 'r', newline='', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        for tweet, label in reader:
            tweets.append(tweet[1:-1]) # remove quotes
            
            labels.append(int(label))    
    return np.array(tweets), np.array(labels)

def create_dictionary(tweets, isBigram = False):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training tweets. Use get_words to process each tweet.

    Only include add words to the dictionary if they occur in at least five tweets.

    Args:
        tweets: A list of strings containing SMS tweets
        isBigram: if true, creates a dictionary of bigrams otherwise, creates a dictionary
        of unigrams

    Returns:
        A python dict mapping words to integers.
    """

    wordCount = {}         # count number of tweets each word appears in 
    wordDict = {}          # final dictionary of words <--> indices to output

    for tweet in tweets:
        wordsSeen = set()  # keep track of whether all unique words in a tweet
        words = tweet.split()
        if isBigram:
            words = [(words[ix],words[ix+1]) for ix in range(0,len(words)-1)]
        words = list(set(words))
        for word in words:
            if word not in wordCount.keys():
                wordCount[word] = 1
            else:
                wordCount[word] = wordCount[word] + 1
            
    # remove all words that are seen less than 5 times from dictionary
    wordCount = {word : num for word, num in wordCount.items() if num >= 5}
             
    # map words that are seen sufficiently to dictionary, and generate index for each word
    i = 0
    for word in wordCount.keys():
        wordDict[word] = i
        i += 1
    
    print('vocab size: '+ str(len(wordDict)))
    return(wordDict)

def transform_text(tweets, word_dictionary, isBigram = False):
    """Transform a list of text tweets into a numpy array for further processing.

    Creates a numpy array that contains the number of times each word
    appears in each tweet. Each row in the resulting array corresponds to each
    tweet and each column corresponds to a word.


    Args:
        tweets: A list of strings where each string is an SMS tweet.
        word_dictionary: A python dict mapping words to integers.
        isBigram: If true, uses bigrams otherwise, uses unigrams

    Returns:
        A numpy array marking the words present in each tweet.
    """

    numtweets = len(tweets)
    numWords = len(word_dictionary.keys())
    tweetArray = np.zeros((numtweets, numWords))
    
    for i in range(len(tweets)):
        words = tweets[i].split()
        if isBigram:
            words = [(words[ix],words[ix+1]) for ix in range(0,len(words)-1)]
        for word in words:
            if word in word_dictionary.keys():
                index = word_dictionary[word]
                tweetArray[i][index] += 1
    
    return(tweetArray)
