# Inspired by https://github.com/tmadl/semisup-learn/blob/82ad804b3f7472d89ae629c3c689c5920edce006/methods/scikitTSVM.py

import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from qns3vm import QN_S3VM
import random as rnd
from itertools import combinations
import util
import pandas as pd

class multiclass_QNS3VM():
    """
    Multiclass wrapper for QN-S3VM
    Args: 
        lam -- regularization parameter lambda (default 1, must be a float > 0)
        lamU -- cost parameter that determines influence of unlabeled patterns (default 1, must be float > 0)
        sigma -- kernel width for RBF kernel (default 1.0, must be a float > 0)
        kernel_type -- "linear" or "rbf" (default "rbf")
        labels -- list of true labels contained in the dataset
    """
    
    def __init__(self, kernel_type = 'rbf', lam = 1e-4, sigma = 0.5, lamU = 1.0,labels = [0,1]):
        self.kernel_type = kernel_type
        self.lam = lam
        self.sigma = sigma
        self.lamU = lamU
        self.random_generator = rnd.Random()
        self.classPairs = list(combinations(labels, 2))
        self.models = {}
        
    def fit(self, X, y): # -1 for unlabeled
        """Fit the model according to the given training data.
        Args: 
            X : numpy array of shape = (n_examples, n_features)
                Training vector, where n_examples in the number of examples and
                n_features is the number of features.
            y : numpy array of shape = (n_examples)
                Target vector relative to X
             Must be one of the values in labels for labeled and -1 for unlabeled instances 
        Returns: None
        """        
        unlabeledX = X[y==-1]

        for c1, c2 in self.classPairs: # perform one-vs-one classification  
            print("training classes "+str(c1)+" vs "+ str(c2))    
            # convert class c1 to 1 and c2 to -1 per svm convention      
            c1X = X[y == c1]
            c1y = np.ones((len(y[y == c1])))

            c2X = X[y == c2]
            c2y = -1*np.ones((len(y[y == c2])))
            
            labeledX = np.concatenate((c1X, c2X), axis = 0)
            labeledy = np.concatenate((c1y, c2y), axis = 0).astype(int)

            if 'rbf' in self.kernel_type.lower():
                self.models[(c1,c2)] = QN_S3VM(labeledX, labeledy, unlabeledX,\
                     lam=self.lam, lam_u=self.lamU, kernel="rbf", sigma=self.sigma)
            else:
                self.models[(c1,c2)] = QN_S3VM(labeledX, labeledy, unlabeledX,  lam=self.lam, lam_u=self.lamU)
                
            self.models[(c1,c2)].fit()
  
    def predict(self, X):
        """Perform classification on examples in X.
        Args:
            X : np array shape = (n_examples, n_features)
        Returns:
            y_pred : np array, shape = (n_examples).Class labels for examples in X.
        """
        preds = []
        for c1, c2 in self.classPairs:
            print("predicting classes "+str(c1)+" vs "+str(c2))    
            y = self.models[c1, c2].get_predictions(X)
            y[y==1] = c1
            y[y==-1] = c2 
            preds.append(y)
        preds = np.array(preds)
        predClass = []

        for j in range(len(preds[0])):
            (values, counts) = np.unique(preds[:,j], return_counts = True)
            ind = np.argmax(counts)
            predClass.append(values[ind])
        predClass[predClass == 2] = -1 # Change back to -1 label
        return predClass

def main():   
    traintweets, trainLabels = util.load_dataset(r'../data/train_new.csv')
    valtweets, valLabels = util.load_dataset(r'../data/val_new.csv')
    testtweets, testLabels = util.load_dataset(r'../data/test_new.csv')

    ## Get unigram bag of words
    wordDict = util.create_dictionary(traintweets, False)
    trainWordMatrix = util.transform_text(traintweets, wordDict, False)
    valWordMatrix = util.transform_text(valtweets, wordDict, False)
    testWordMatrix = util.transform_text(testtweets, wordDict, False)
    
    trainNeg = trainWordMatrix[trainLabels == -1]
    trainNeut = trainWordMatrix[trainLabels == 0]
    trainPos = trainWordMatrix[trainLabels == 1]
    m, n = trainPos.shape
    posIdx = np.random.choice( m, int(m/2), replace= False)
    trainPos = trainPos[posIdx]

    trainLabels = np.concatenate((-1*np.ones((len(trainNeg))), np.zeros((len(trainNeut))), np.ones(len(posIdx))))

    trainWordMatrix = np.concatenate((trainNeg, trainNeut, trainPos), axis=0)
    # change all -1 labels to 2 since -1 reserved for unlabelled
    trainLabels[trainLabels == -1] = 2 
    valLabels[valLabels == -1] = 2
    testLabels[testLabels == -1] = 2

    txt_path = r'../data/unlabelled3_06.txt'
    unlabelledtweets = util.load_unlabelled_dataset(txt_path)
    unlabelledMatrix = util.transform_text(unlabelledtweets, wordDict, False)
    unlabelledy = -1 * np.ones((len(unlabelledMatrix)))

    # concatenate into single feature and label matrices
    X = np.concatenate((trainWordMatrix, unlabelledMatrix), axis=0)    
    y = np.concatenate((trainLabels, unlabelledy), axis = 0)

    model = multiclass_QNS3VM(labels = [0, 1, 2])
    print("fitting" )
    model.fit(X, y)
    print("predicting")
    preds_train = model.predict(trainWordMatrix)
    trainLabels[trainLabels == 2] = -1 
    print('S3VM train accuracy: '+str(np.mean(trainLabels == preds_train)))

    valLabels[valLabels == 2 ] = -1
    preds_val = model.predict(valWordMatrix)
    print('S3VM val accuracy: '+str(np.mean(valLabels == preds_val)))

    # Plot confusion matrix
    y_actual = pd.Series(valLabels, name='Actual')
    y_pred = pd.Series(preds_val, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 's3svm_val_us')

    # Plot confusion matrix
    testLabels[testLabels == 2] = -1
    preds_test = model.predict(testWordMatrix)
    print('S3VM test accuracy: '+str(np.mean(testLabels == preds_test)))
    y_actual = pd.Series(testLabels, name='Actual')
    y_pred = pd.Series(preds_test, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 's3svm_test_us')

if __name__ == '__main__':
    main()
    
