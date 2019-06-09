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
    Multiclass wrapper for QN-S3VM by Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer (see http://www.fabiangieseke.de/index.php/code/qns3vm) 
    and provides probability estimates using Platt scaling.
    Parameters
    ----------
    lam -- regularization parameter lambda (default 1, must be a float > 0)
    lamU -- cost parameter that determines influence of unlabeled patterns (default 1, must be float > 0)
    sigma -- kernel width for RBF kernel (default 1.0, must be a float > 0)
    kernel_type -- "linear" or "rbf" (default "rbf")
    probability -- whether to enable probability estimates. (boolean, optional, default=False)  
    """
    
    def __init__(self, kernel_type = 'rbf', lam = 1e-4, sigma = 0.5, lamU = 1.0, probability=False, labels = [0,1]):
        self.kernel_type = kernel_type
        self.lam = lam
        self.sigma = sigma
        self.lamU = lamU
        self.probability = probability
        self.random_generator = rnd.Random()
        self.plattlr = None
        self.classPairs = list(combinations(labels, 2))
        self.models = {}
        
    def fit(self, X, y): # -1 for unlabeled
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : numpy array of shape = (n_examples, n_features)
            Training vector, where n_examples in the number of examples and
            n_features is the number of features.
        y : numpy array of shape = (n_examples)
            Target vector relative to X
            Must be one of the values in labels for labeled and -1 for unlabeled instances 
        labels : list of labels corresponding to the different classes
        Returns
        -------
        self : object
            Returns self.
        """        
        unlabeledX = X[y==-1]

        for c1, c2 in self.classPairs: # perform one-vs-one classification  
            print(f"training classes {c1} vs {c2}")    
            # convert class c1 to 1 and c2 to -1 per svm convention      
            c1X = X[y == c1]
            c1y = np.ones((len(y[y == c1])))

            c2X = X[y == c2]
            c2y = -1*np.ones((len(y[y == c2])))
            
            labeledX = np.concatenate((c1X, c2X), axis = 0)
            labeledy = np.concatenate((c1y, c2y), axis = 0).astype(int)

            #if 'rbf' in self.kernel_type.lower():
                #self.models[(c1,c2)] = QN_S3VM(labeledX, labeledy, unlabeledX,\
                     #lam=self.lam, lam_u=self.lamU, kernel="rbf", sigma=self.sigma)
            if 'rbf' in self.kernel_type.lower():
                self.models[(c1,c2)] = QN_S3VM(labeledX, labeledy, unlabeledX, random_generator = self.random_generator,\
                     lam=self.lam, lam_u=self.lamU, kernel="rbf", sigma=self.sigma)
            else:
                self.models[(c1,c2)] = QN_S3VM(labeledX, labeledy, unlabeledX,  lam=self.lam, lam_u=self.lamU)
                
            self.models[(c1,c2)].train()
            
            # probabilities by Platt scaling
            if self.probability:
                self.plattlr = LR()
                preds = self.models[(c1, c2)].getPredictions(labeledX, real_valued=True)
                self.plattlr.fit(preds.reshape( -1, 1 ), labeledy)
        
    def predict_proba(self, X, classPair):
        """Compute probabilities of possible outcomes for samples in X.
        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.
        Parameters
        ----------
        X : numpy array of shape = (n_examples, n_features)
            Training vector, where n_examples in the number of examples and
            n_features is the number of features.
        -------
        T : array-like, shape = (n_examples, n_classes)
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        prob = []

        for c1, c2 in self.classPairs:
            if self.probability:
                preds = self.models[classPair].getPredictions(X, real_valued=True)
                prob.append(self.plattlr.predict_proba(preds.reshape( -1, 1 )))
            else:
                raise RuntimeError("Probabilities were not calculated for this model - make sure you pass probability=True to the constructor")
        
        predClass = np.argmax(np.array(prob), axis = 0)
        return [self.classPairs[pred][0] if prob[pred][i] > 0.5 else self.classPairs[pred][1] for i, pred in enumerate(predClass)]
    
    def predict(self, X):
        """Perform classification on examples in X.
        Parameters
        ----------
        X : np array shape = (n_examples, n_features)
        Returns
        -------
        y_pred : np array, shape = (n_examples)
            Class labels for examples in X.
        """
        preds = []
        for c1, c2 in self.classPairs:
            print(f"predicting classes {c1} vs {c2}")    
            y = self.models[c1, c2].getPredictions(X)
            y[y==1] = c1
            y[y==-1] = c2 
            preds.append(y)

        predClass = [max(preds[:,i],key=preds[:,i].count) for i in range(len(preds[0]))]
        
        return predClass

def main():   
    traintweets, trainLabels = util.load_dataset(r'..\data\train_new.csv')
    valtweets, valLabels = util.load_dataset(r'..\data\val_new.csv')
    testtweets, testLabels = util.load_dataset(r'..\data\test_new.csv')
    #traintweets, trainLabels = util.load_dataset(r'data\train_new.csv')
    #valtweets, valLabels = util.load_dataset(r'data\val_new.csv')
    #testtweets, testLabels = util.load_dataset(r'data\test_new.csv')

    ## Get unigram bag of words
    wordDict = util.create_dictionary(traintweets, False)
    trainWordMatrix = util.transform_text(traintweets, wordDict, False)
    valWordMatrix = util.transform_text(valtweets, wordDict, False)
    testWordMatrix = util.transform_text(testtweets, wordDict, False)


    # change all -1 labels to 2 since -1 reserved for unlabelled
    trainLabels[trainLabels == -1] = 2 
    valLabels[valLabels == -1] = 2
    testLabels[testLabels == -1] = 2

    txt_path = r'..\data\unlabelled3_06.txt'
    #txt_path = r'data\unlabelled3_06.txt'
    unlabelledtweets = util.load_unlabelled_dataset(txt_path)
    unlabelledMatrix = util.transform_text(unlabelledtweets, wordDict, False)
    unlabelledy = -1 * np.ones((len(unlabelledMatrix)))

    # concatenate into single feature and label matrices
    X = np.concatenate((trainWordMatrix, unlabelledMatrix), axis=0)    
    y = np.concatenate((trainLabels, unlabelledy), axis = 0)
    
    model = multiclass_QNS3VM(labels = [0, 1, 2])
    print("Training model")
    model.fit(X, y)
    print("predicting")
    preds_val = model.predict(valWordMatrix)
    print(f'S3VM val accuracy: {np.mean(valLabels == preds_val)}')


    # Plot confusion matrix
    y_actual = pd.Series(valLabels, name='Actual')
    y_pred = pd.Series(preds_val, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 's3svm val')

    print("predicting val")
    preds_val = model.predict(valWordMatrix)
    print(f'S3VM val accuracy: {np.mean(valLabels == preds_val)}')


    # Plot confusion matrix
    print("predicting test")
    preds_test = model.predict(testWordMatrix)
    print(f'S3VM test accuracy: {np.mean(valLabels == preds_test)}')
    y_actual = pd.Series(testLabels, name='Actual')
    y_pred = pd.Series(preds_test, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred, title = 's3svm test')

if __name__ == '__main__':
    main()
    