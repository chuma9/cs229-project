import collections
import numpy as np
import csv
from tqdm import tqdm
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt
import pandas as pd
import util
from util import *

def fit_naive_bayes_model_3(matrix, labels):
    """Fit a naive bayes model.

    This function fits a Naive Bayes model given a training matrix and labels.

    Args:
        matrix: A numpy array containing word counts for the training data. Assume first n columns
            (corresponding to first n labels) are labeled examples, and that all remaining columns
            are unlabelled examples.
        labels: The multivariate (-1, 0, 1) labels for that training data

    Returns: The trained model
    """
    n = len(labels)     # number of labeled tweets present
    m = matrix.shape[0] - n   # number of unlabeled tweets present
    v = matrix.shape[1] # size of vocabulary
    labelTypes = [-1, 0, 1] # types of labels in our data
    
    theta_y = []
    theta_k = []
    
    # "warm up" estimation for theta_y by using predictions from supervised learning
    for lab in labelTypes:
        fracDataWithLab = np.sum(labels == lab) / n
        theta_y.append(fracDataWithLab)
    
    # "warm up" estimation for theta_k given each value of y, store as array of length v = size of vocab
    for lab in labelTypes:
        examples = matrix[labels == lab, :]
        sumExamples = np.sum(examples)
        prob = (1 + np.sum(examples, axis=0)) / (sumExamples + v) # assign MLE with laplace smoothing
        theta_k.append(prob)
    
    theta_k = np.stack(theta_k, axis=0)
    return(np.array(theta_y), theta_k)

def fit_semisupervised_naive_bayes_model(matrix, labels):
    """Fit a semisupervised naive bayes model.

    This function fits a Naive Bayes model given a training matrix and labels.

    Args:
        matrix: A numpy array containing word counts for the training data. Assume first n columns
            (corresponding to first n labels) are labeled examples, and that all remaining columns
            are unlabelled examples.
        labels: The multivariate (-1, 0, 1) labels for that training data

    Returns: The trained model
    """
    
    # hyperparameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-2   # Convergence threshold
    max_iter = 1000

    # data parameters
    n = len(labels)     # number of labeled tweets present
    m = matrix.shape[0] - n   # number of unlabeled tweets present
    v = matrix.shape[1] # size of vocabulary
    labelTypes = np.array([-1, 0, 1]) # types of labels in our data
    numLabels = labelTypes.shape[0]
    
    labelledMatrix = matrix[:n, :]
    unlabelledMatrix = matrix[n:, :]
    
    theta_y = []
    theta_k = []
    
    # "warm up" estimation for theta_y by using predictions from supervised learning
    for lab in labelTypes:
        fracDataWithLab = np.sum(labels == lab) / n
        theta_y.append(fracDataWithLab)
    
    # "warm up" estimation for theta_k given each value of y, store as array of length v = size of vocab
    for lab in labelTypes:
        examples = labelledMatrix[labels == lab, :]
        sumExamples = np.sum(examples)
        prob = (1 + np.sum(examples, axis=0)) / (sumExamples + v) # assign MLE with laplace smoothing
        theta_k.append(prob)
    
    theta_k = np.stack(theta_k, axis=0)

    # Run EM algorithm with semisupervised data
    # Stop when the absolute change in log-likelihood is < eps
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # E-step: Update qis
        qis = np.zeros((m, numLabels))
        for i in range(numLabels):
            qis[:, i] = np.exp(np.dot(unlabelledMatrix, np.log(theta_k[i])) + np.log(theta_y[i]))
        qis = (qis.T / np.sum(qis, 1)).T

        # (2) M-step: Update the model parameters theta_k and theta_y
        for i in range(numLabels):
            fracLabeledData = np.sum(labels == labelTypes[i])
            weightsUnlabeledData = np.sum(qis[:, i])
            updatedThetaY = (1+ alpha*fracLabeledData + weightsUnlabeledData) / (numLabels + alpha*n + m)
            theta_y[i] = updatedThetaY
            
        for i in range(numLabels):
            examples = labelledMatrix[labels == labelTypes[i], :]
            sumExamples = np.sum(examples)
            
            unlabeledExamples = np.matmul(unlabelledMatrix.T, qis[:, i])
            sumUnlabeledExamples = np.sum(unlabeledExamples)
            prob = (1 + alpha*np.sum(examples, axis=0) + unlabeledExamples) / (alpha*sumExamples + sumUnlabeledExamples + v) # assign MLE with laplace smoothing
            theta_k[i, :] = prob

        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = 0
        labeledDataLoss = 0
        unlabeledDataLoss = 0
        for i in range(numLabels):
            labeledDataLoss += alpha*np.sum(np.matmul(labelledMatrix[labels == labelTypes[i]], np.log(theta_k[i, :])))
            labeledDataLoss += alpha*np.sum(labels == labelTypes[i])*np.log(theta_y[i])

        unlabeledExampleLoss = np.exp(np.dot(unlabelledMatrix, np.log(theta_k).T) + np.log(theta_y[i]))
        unlabeledExampleLoss = np.sum(unlabeledExampleLoss, axis=1)
        unlabeledDataLoss = np.sum(np.log(unlabeledExampleLoss))
        
        ll += labeledDataLoss + unlabeledDataLoss
        
        it += 1
        print('[iter: {:03d}, log-likelihood: {:.4f}]'.format(it, ll))

    return(theta_y, theta_k)

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

    for i in range(numClasses):
        examplesLogProb = np.dot(matrix, logProbK[i])
        exampleProb = np.exp(examplesLogProb + logProbY[i])
        exampleProbs.append(exampleProb)
    
    totalProbs = np.zeros(exampleProbs[0].shape)
    for i in range(numClasses):
        totalProbs += exampleProbs[i]
    
    dataLabels = np.zeros(matrix.shape[0])
    maxExampleProbs = np.zeros(totalProbs.shape)
    for i in range(numClasses):
        prob = exampleProbs[i] / totalProbs
        dataLabels[prob > maxExampleProbs] = labelTypes[i]
        maxExampleProbs[prob > maxExampleProbs] = prob[prob > maxExampleProbs]
        
    return(dataLabels)

def self_learn(trainMatrix, trainLabels, unlabelledMatrix, maxIter = 20, threshold = 0.50):
    ''' Update training data with confident predictions from unlabelled data
    '''
    original_labelled = len(trainMatrix)

    for i in tqdm(range(maxIter)): # self learn for maxIter iterations
        NBModel = fit_naive_bayes_model_3(trainMatrix, trainLabels)
        preds = learn_from_naive_bayes_model_3(NBModel, unlabelledMatrix, threshold=threshold, len_og_labelled = original_labelled)

        # add to training examples
        posNewLabels = unlabelledMatrix[preds == 1]
        negNewLabels = unlabelledMatrix[preds == -1]
        neutNewLabels = unlabelledMatrix[preds == 0]

        trainMatrix = np.concatenate((trainMatrix,posNewLabels, negNewLabels, neutNewLabels), axis=0)
        trainLabels = np.concatenate((trainLabels,np.ones((len(posNewLabels))), \
            -1*np.ones((len(negNewLabels))), np.zeros((len(neutNewLabels)))), axis=0)     

        if np.array_equal(unlabelledMatrix, unlabelledMatrix[preds == 5]): # stop. no confident predictions in the unlabelled set
            break
        unlabelledMatrix = unlabelledMatrix[preds == 5]

    return fit_naive_bayes_model_3(trainMatrix, trainLabels)


def learn_from_naive_bayes_model_3(model, matrix, threshold = 0.5, len_og_labelled = None):
    """Use a Naive Bayes model to compute predictions for a unlabelled data matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts
        threshold: Confidence threshold used for labelling new data during self learinng

    Returns: A numpy array containg the confident predictions from the model 
    """
    if len_og_labelled is None:
        raise RuntimeError("The length of the original set of labelled training examples must be set")
    theta_y, theta_k = model
    numClasses = len(theta_y)
    labelTypes = [-1, 0, 1]
    exampleProbs = []

    logProbY = np.log(theta_y)
    logProbK = np.log(theta_k)

    for i in range(numClasses):
        examplesLogProb = np.dot(matrix, logProbK[i])
        exampleProb = np.exp(examplesLogProb + logProbY[i])
        exampleProbs.append(exampleProb)
    
    totalProbs = np.zeros(exampleProbs[0].shape)
    for i in range(numClasses):
        totalProbs += exampleProbs[i]
    
    dataLabels = np.zeros(matrix.shape[0])
    maxExampleProbs = np.zeros(totalProbs.shape)
    for i in range(numClasses):
        prob = exampleProbs[i] / totalProbs
        dataLabels[prob > maxExampleProbs] = labelTypes[i]
        maxExampleProbs[prob > maxExampleProbs] = prob[prob > maxExampleProbs]
    # self train
    for i in range(len_og_labelled, len(maxExampleProbs)): # skip the original training data
        if maxExampleProbs[i] < threshold: # only label values with probability > threshold
            dataLabels[i] = 5 # use 5 to indicate no label

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


def main():

    ##### CODE EXAMPLEs ######
    trainTweets, trainLabels = load_dataset(r'../data/2016_train.csv')
    valTweets, valLabels = load_dataset(r'../data/2016_val.csv')
    testTweets, testLabels = load_dataset(r'../data/2016_test.csv')
    unlabelledTweets = load_unlabelled_dataset(r'../data/unlabelled_dataset.txt')
    
    ## Train EM unigram model
    import pdb; pdb.set_trace()
        
    combinedTweets = np.append(trainTweets, unlabelledTweets)
    wordDict = create_dictionary(combinedTweets, False)
    combinedMatrix = transform_text(combinedTweets, wordDict, False)
    trainMatrix = transform_text(trainTweets, wordDict)
    NBModel = fit_semisupervised_naive_bayes_model(combinedMatrix, trainLabels)
    # NBModel = fit_naive_bayes_model_3(trainMatrix, trainLabels)
    
    trainWordMatrix = transform_text(trainTweets, wordDict, False)
    trainPreds = predict_from_naive_bayes_model_3(NBModel, trainWordMatrix)
    trainAcc = np.mean(trainPreds == trainLabels)

    valWordMatrix = transform_text(valTweets, wordDict, False)
    valPreds = predict_from_naive_bayes_model_3(NBModel, valWordMatrix)
    valAcc = np.mean(valPreds == valLabels)

    testWordMatrix =  transform_text(testTweets, wordDict, False)
    testPreds = predict_from_naive_bayes_model_3(NBModel, testWordMatrix)
    testAcc = np.mean(testPreds == testLabels)

    y_actual = pd.Series(testLabels, name='Actual')
    y_pred = pd.Series(testPreds, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 'unigram_test')

    
    # resultsFile = '../data/results_full.csv'
    # with open(resultsFile, 'w+') as fp:
    #     fp.write(f'{trainAcc}, {valAcc}, {testAcc}\n')

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


    
    traintweets, trainLabels = util.load_dataset(r'..\data\train_new.csv')
    valtweets, valLabels = util.load_dataset(r'..\data\val_new.csv')
    testtweets, testLabels = util.load_dataset(r'..\data\test_new.csv')

    ## Train unigram model
    wordDict = util.create_dictionary(traintweets, False)
    trainWordMatrix = util.transform_text(traintweets, wordDict, False)

    ## Train bigram model
    bigramDict = util.create_dictionary(traintweets, True)
    trainBigramMatrix = util.transform_text(traintweets, bigramDict, True)

    valWordMatrix = util.transform_text(valtweets, wordDict, False)
    valBigramMatrix = util.transform_text(valtweets, bigramDict, True)

    testWordMatrix = util.transform_text(testtweets, wordDict, False)
    testBigramMatrix = util.transform_text(testtweets, bigramDict, True)

    # Predict Unigram model
    NBModel = fit_naive_bayes_model_3(trainWordMatrix, trainLabels)
    preds_train = predict_from_naive_bayes_model_3(NBModel, trainWordMatrix)
    preds_val = predict_from_naive_bayes_model_3(NBModel, valWordMatrix)
    preds_test = predict_from_naive_bayes_model_3(NBModel, testWordMatrix)

    print(f'Unigram train accuracy: {np.mean(trainLabels == preds_train)}')
    print(f'Unigram val accuracy: {np.mean(valLabels == preds_val)}')
    print(f'Unigram test accuracy: {np.mean(testLabels == preds_test)}')

    # Plot confusion matrix
    y_actual = pd.Series(trainLabels, name='Actual')
    y_pred = pd.Series(preds_train, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 'unigram_train')

    y_actual = pd.Series(valLabels, name='Actual')
    y_pred = pd.Series(preds_val, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 'unigram_val')

    y_actual = pd.Series(testLabels, name='Actual')
    y_pred = pd.Series(preds_test, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 'unigram_test')


    # Predict Bigram model

    NBModel = fit_naive_bayes_model_3(trainBigramMatrix, trainLabels)
    preds_val = predict_from_naive_bayes_model_3(NBModel, valBigramMatrix)
    preds_train = predict_from_naive_bayes_model_3(NBModel, trainBigramMatrix)
    preds_test = predict_from_naive_bayes_model_3(NBModel, testBigramMatrix)
    print(f'Bigram train accuracy: {np.mean(trainLabels == preds_train)}')
    print(f'Bigram val accuracy: {np.mean(valLabels == preds_val)}')
    print(f'Bigram test accuracy: {np.mean(testLabels == preds_test)}')


    # Plot confusion matrix
    y_actual = pd.Series(trainLabels, name='Actual')
    y_pred = pd.Series(preds_train, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 'bigram_train')

    y_actual = pd.Series(valLabels, name='Actual')
    y_pred = pd.Series(preds_val, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 'bigram_val')

    y_actual = pd.Series(testLabels, name='Actual')
    y_pred = pd.Series(preds_test, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 'bigram_test')

    # IF SELF LEARNING
    txt_path = r'..\data\unlabelled3_06.txt'
    unlabelledtweets = util.load_unlabelled_dataset(txt_path)
    unlabelledMatrix = util.transform_text(unlabelledtweets, wordDict, False)
    NBModel = self_learn(trainWordMatrix, trainLabels, unlabelledMatrix)
    preds_train = predict_from_naive_bayes_model_3(NBModel, trainWordMatrix)
    print(f'Self Learning val accuracy: {np.mean(trainLabels == preds_train)}')

    preds_val = predict_from_naive_bayes_model_3(NBModel, valWordMatrix)
    print(f'Self Learning val accuracy: {np.mean(valLabels == preds_val)}')

    preds_test = predict_from_naive_bayes_model_3(NBModel, testWordMatrix)
    print(f'Self Learning test accuracy: {np.mean(testLabels == preds_test)}')


if __name__ == '__main__':
    main()
