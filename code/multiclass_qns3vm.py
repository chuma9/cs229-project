# Inspired by https://github.com/tmadl/semisup-learn/blob/82ad804b3f7472d89ae629c3c689c5920edce006/methods/scikitTSVM.py

import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from qns3vm import QN_S3VM
import random as rnd
from itertools import combinations

class multiclass():
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
            # convert class c1 to 1 and c2 to -1 per svm convention      
            c1X = X[y == c1]
            c1y = np.ones(len(y[y == c1]), 1)

            c2X = X[y == c2]
            c2y = -1*np.ones(len(y[y == c2]),1)
            
            labeledX = np.concatenate((c1X, c2X), axis = 0)
            labeledy = np.concatenate((c1y, c2y), axis = 0)

            if 'rbf' in self.kernel_type.lower():
                self.models[(c1,c2)] = QN_S3VM(labeledX.tolist(), labeledy, unlabeledX, self.random_generator,\
                     lam=self.lam, lamU=self.lamU, kernel_type="RBF", sigma=self.sigma)
            else:
                self.models[(c1,c2)] = QN_S3VM(labeledX.tolist(), labeledy, unlabeledX, self.random_generator, lam=self.lam, lamU=self.lamU)
                
            self.models[(c1,c2)].train()
            
            # probabilities by Platt scaling
            if self.probability:
                self.plattlr = LR()
                preds = self.models[(c1, c2)].getPredictions(labeledX.tolist(), real_valued=True)
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
                preds = self.models[classPair].getPredictions(X.tolist(), real_valued=True)
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
            y = self.models[c1, c2].getPredictions(X.tolist())
            y[y==1] = c1
            y[y==-1] = c2 
            preds.append(y)

        predClass = [max(preds[:,i],key=preds[:,i].count) for i in range(len(preds[0]))]
        
        return predClass

def main():   
    traintweets, trainLabels = util.load_dataset(r'data\train_new.csv')
    valtweets, valLabels = util.load_dataset(r'data\val_new.csv')

    ## Get unigram bag of words
    wordDict = util.create_dictionary(traintweets, False)
    trainWordMatrix = util.transform_text(traintweets, wordDict, False)
    valWordMatrix = util.transform_text(valtweets, wordDict, False)


    print(f'Unigram accuracy: {np.mean(valLabels == preds)}')

    # Plot confusion matrix
    y_actual = pd.Series(valLabels, name='Actual')
    y_pred = pd.Series(preds, name='Predicted')
    util.plot_confusion_matrix(y_actual, y_pred)

if __name__ == '__main__':
    main()
    