import collections
import numpy as np
import csv
from tqdm import tqdm
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt
import pandas as pd
import util

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

def main():
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
>>>>>>> d4055cd25fa695b8499e9ab6ffc38d036b343608
