import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        weights = [1.0/len(trainingData)]*len(trainingData)

        for k in range(self.boosting_iterations):
            self.classifiers[k].train(trainingData ,trainingLabels, weights)
            error=0.0
            M = self.classifiers[k].classify(trainingData)
            for i in range(len(trainingData)):
                if M[i] != trainingLabels[i]:
                    error+=weights[i]
            # print("error", error)
            for i in range(len(trainingData)):
                if M[i] == trainingLabels[i]:
                    weights[i] *= error/(1-error)
            weights = util.normalize(weights)
            # print(weights)
            self.alphas[k]=np.log((1-error)/error)

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """
        X = [util.Counter() for _ in data]
        labels = [0]*len(data)
        for k in range(self.boosting_iterations):
            M = self.classifiers[k].classify(data)
            for i in range(len(data)):
                X[i][M[i]] +=self.alphas[k]
        return [x.argMax() for x in X]
