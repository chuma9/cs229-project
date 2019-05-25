import collections
import numpy as np
import csv
from tqdm import tqdm
from nltk.corpus import stopwords 

def load_dataset(csv_path):
    """Load a CSV file containing a dataset and return a numpy array of tweets and another of corresponding labels

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

def load_unlabelled_dataset(txt_path):
    tweets = []

    with open(txt_path, 'r', encoding='utf8') as txt_file:
        for tweet in txt_file:
            tweets.append(tweet[1:-1]) # remove quotes
    return np.array(tweets)
   
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
    stopWords = set(stopwords.words("english"))

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
            
    # remove all words that are seen less than 5 times and all stop words from dictionary
    wordCount = {word : num for word, num in wordCount.items() if num >= 5 and word not in stopWords}
            
    # map words that are seen sufficiently to dictionary, and generate index for each word
    i = 0
    for word in wordCount.keys():
        wordDict[word] = i
        i += 1
        
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

def fit_naive_bayes_model_3(matrix, labels):
    """Fit a naive bayes model.

    This function fits a Naive Bayes model given a training matrix and labels.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The multivariate (-1, 0, 1) labels for that training data

    Returns: The trained model
    """
    n = len(labels)     # number of tweets present
    v = matrix.shape[1] # size of vocabulary
    labelTypes = [-1, 0, 1] # types of labels in our data
    
    theta_y = []
    theta_k = []
    
    # first calculate theta_y
    for lab in labelTypes:
        fracDataWithLab = np.sum(labels == lab) / n
        theta_y.append(fracDataWithLab)
    
    # calculate theta_k given each value of y, store as array of length v = size of vocab
    for lab in labelTypes:
        examples = matrix[labels == lab, :]
        sumExamples = np.sum(examples)
        prob = (1 + np.sum(examples, axis=0)) / (sumExamples + v) # assign MLE with laplace smoothing
        theta_k.append(prob)
    
    theta_k = np.stack(theta_k, axis=0)
    return(np.array(theta_y), theta_k)

def predict_from_naive_bayes_model_3(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    theta_y, theta_k = model
    numClasses = len(theta_y)
    labelTypes = [-1, 0, 1]
    exampleProbs = []
    
    logProbY = np.log(theta_y)
    logProbK = np.log(theta_k)

    for i in range(len(theta_y)):
        examplesLogProb = np.dot(matrix, logProbK[i])
        exampleProb = np.exp(examplesLogProb + logProbY[i])
        exampleProbs.append(exampleProb)
    
    totalProbs = np.zeros(exampleProbs[0].shape)
    for i in range(len(theta_y)):
        totalProbs += exampleProbs[i]
    
    dataLabels = np.zeros(matrix.shape[0])
    maxExampleProbs = np.zeros(totalProbs.shape)
    for i in range(len(theta_y)):
        prob = exampleProbs[i] / totalProbs
        dataLabels[prob > maxExampleProbs] = labelTypes[i]
        maxExampleProbs[prob > maxExampleProbs] = prob[prob > maxExampleProbs]
        
    return(dataLabels)

def self_learn(trainMatrix, trainLabels, unlabelledMatrix):
    ''' Update training data with confident predictions from unlabelled data
    '''

    for i in tqdm(range(10)): # self learn for 10 iterations
        NBModel = fit_naive_bayes_model_3(trainMatrix, trainLabels)
        preds = learn_from_naive_bayes_model_3(NBModel, unlabelledMatrix)

        # add to training examples
        # Don't learn pro, already overrepresented
        #posNewLabels = unlabelledMatrix[preds == 1]
        #trainMatrix = np.concatenate((trainMatrix,posNewLabels), axis=0)
        #trainLabels = np.concatenate((trainLabels,np.array([1]*len(posNewLabels))), axis=0)
        negNewLabels = unlabelledMatrix[preds == -1]
        trainMatrix = np.concatenate((trainMatrix,negNewLabels), axis=0)
        trainLabels = np.concatenate((trainLabels,np.array([-1]*len(negNewLabels))), axis=0)

        
        neutNewLabels = unlabelledMatrix[preds == 0]
        trainMatrix = np.concatenate((trainMatrix,neutNewLabels), axis=0)
        trainLabels = np.concatenate((trainLabels,np.array([0]*len(neutNewLabels))), axis=0)
        unlabelledMatrix = unlabelledMatrix[preds == 5]

    return fit_naive_bayes_model_3(trainMatrix, trainLabels)


def learn_from_naive_bayes_model_3(model, matrix):
    """Use a Naive Bayes model to compute predictions for a unlabelled data matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the confident predictions from the model 
    """
    theta_y, theta_k = model
    numClasses = len(theta_y)
    labelTypes = [-1, 0, 1]
    exampleProbs = []
    
    logProbY = np.log(theta_y)
    logProbK = np.log(theta_k)

    for i in range(len(theta_y)):
        examplesLogProb = np.dot(matrix, logProbK[i])
        exampleProb = np.exp(examplesLogProb + logProbY[i])
        exampleProbs.append(exampleProb)
    
    totalProbs = np.zeros(exampleProbs[0].shape)
    for i in range(len(theta_y)):
        totalProbs += exampleProbs[i]
    
    dataLabels = np.zeros(matrix.shape[0])
    maxExampleProbs = np.zeros(totalProbs.shape)
    for i in range(len(theta_y)):
        prob = exampleProbs[i] / totalProbs
        dataLabels[prob > maxExampleProbs] = labelTypes[i]
        maxExampleProbs[prob > maxExampleProbs] = prob[prob > maxExampleProbs]
    # self train
    for i in range(len(maxExampleProbs)):
        if maxExampleProbs[i] < 0.98: # only label values with probability > 0.98
            dataLabels[i] = 5 # user 5 to indicate no label

    return(dataLabels)

def get_top_naive_bayes_words(model, dictionary, n=10):
    """Compute the top five words that are most indicative of each class.

    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of lists of the top five most indicative words for each class in sorted 
        order with the most indicative first
    """
    theta_y, theta_k = model
    revDict = {val : key for key, val in dictionary.items()}
    words = []
    
    for i in range(len(theta_y)):
        diff = theta_k[i] / (1 - theta_k[i])
        sortedIds = np.argsort(diff)
        sortedIds = np.flip(sortedIds)

        words.append([revDict[i] for i in sortedIds[0:n]])

    return(words)

##### CODE EXAMPLEs ######
traintweets, trainLabels = load_dataset(r'../data/2016_train.csv')
valtweets, valLabels = load_dataset(r'../data/2016_val.csv')

## Train unigram model
wordDict = create_dictionary(traintweets, False)
trainWordMatrix = transform_text(traintweets, wordDict, False)

## Train bigram model
#bigramDict = create_dictionary(traintweets, True)
#trainBigramMatrix = transform_text(traintweets, bigramDict, True)

valWordMatrix = transform_text(valtweets, wordDict, False)
#valBigramMatrix = transform_text(valtweets, bigramDict, True)

# IF USING 3 CLASSES (NEG/POS/NEUTRAL)
# Predict Unigram model
NBModel = fit_naive_bayes_model_3(trainWordMatrix, trainLabels)
preds = predict_from_naive_bayes_model_3(NBModel, valWordMatrix)
print(f'Unigram accuracy: {np.mean(valLabels == preds)}')

# Predict Bigram model
'''
NBModel = fit_naive_bayes_model_3(trainBigramMatrix, trainLabels)
preds = predict_from_naive_bayes_model_3(NBModel, valBigramMatrix)
print(f'Bigram accuracy: {np.mean(valLabels == preds)}')
''' 

# IF SELF LEARNING
txt_path = r'../data/unlabelled_dataset.txt'
unlabelledtweets = load_unlabelled_dataset(txt_path)
unlabelledMatrix = transform_text(unlabelledtweets, wordDict, False)
NBModel = self_learn(trainWordMatrix, trainLabels, unlabelledMatrix)
preds = predict_from_naive_bayes_model_3(NBModel, valWordMatrix)
print(f'Self Learning accuracy: {np.mean(valLabels == preds)}')
import pdb; pdb.set_trace()