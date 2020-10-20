'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the conservative algorithm selection, only based on historical occurences and a simple metric that weights score vs. time.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 
import numpy as np
### custom (local) imports ### 
from .base.framework import AlgorithmSelector
from .base.framework import OutOfTimeException


class ConservativeAlgorithmSelector(AlgorithmSelector):
    '''
        algorithms - vector of algorithms to run
        callback - callback function that takes: algorithm, problem -> score, cost (budget cost)
            
    '''
    def __init__(self, algorithms):
        self._algorithms = algorithms
        self._noOfAlgorithms = np.shape(self._algorithms)[0]
        self._performancesum = np.zeros(self._noOfAlgorithms)
        self._runs = np.zeros(self._noOfAlgorithms)
        self._performance = np.zeros(self._noOfAlgorithms)
        #fix set the internal hyperparameters. They are quite robust and only change the behaviour if set to extreme values.
        self._explorationFactor = 1.0 #how much weight we put in exploration
        self._explorationDegrationFactor = 1.0 #hyperparameter between 0 and 1

    '''
        run the algorithm selection on one problem instance
    '''
    def process(self, budget, callback):
        #print("past performances: "+str(self._performance))
        #print("calls: "+str(self._runs))
        #set of the indexes of the remaining algorithms
        algorithms = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        #to record the scores and costs of already run algorithms
        scores = np.zeros(np.shape(self._algorithms))
        costs = np.zeros(np.shape(self._algorithms))
        #array that records if an algorithm has been run
        runs = np.zeros(np.shape(self._algorithms), dtype=bool)
        remainingBudget = budget
        regularization = 10**-5

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            remainingPerformances = self._performance[algorithms]
            
            #Choose the algorithm randomly with a probability proportionally to it's historical performance
            
            probabilities = remainingPerformances + self._explorationFactor * ( (np.sum(remainingPerformances)==0) + np.linalg.norm(remainingPerformances, ord=1) / len(remainingPerformances) )
            
            normOfProbabilities = np.linalg.norm(probabilities, ord=1)
            if ( normOfProbabilities > 0 ):
                probabilities = probabilities / normOfProbabilities
            else:
                probabilities[:] = 1

            #print(probabilities)
            selectedAlgorithm = np.random.choice(a = algorithms, replace = False, p = probabilities)

            #remove the selected algorithm from the local set:
            algorithms = algorithms[algorithms != selectedAlgorithm]
            #run the algorithm and get the budget and cost back
            try:
                ( cost, score ) = callback( self._algorithms[ selectedAlgorithm ])
            except OutOfTimeException:
                cost = remainingBudget
                score = 0
            
            #if we exceeded the budget, this run does not count.
            if (remainingBudget < cost ):
                cost = remainingBudget
                score = 0
                
            remainingBudget -= cost
            #print("algorithm:"+str(self._algorithms[ selectedAlgorithm ])+" score: "+str(score)+" cost: "+str(cost))
            runs[selectedAlgorithm] = True
            
            costs[selectedAlgorithm] = cost
            scores[selectedAlgorithm] = score
            
        costs += regularization #fix against zero costs
        if ( np.max(scores)!=0 ) & ( np.max(costs) != 0 ):
            #calculate relative scores and costs
            scores /= np.max(scores)
            costs /= np.max(costs)
            #update meta-learning knowledge
            
            #TODO: Replace with A3R from: S.M. Abdulrahman et.Al. - Measures for Combining Accuracy and Time for Metalearning
            performance = scores[runs] / costs[runs]
            
            self._performancesum[runs] += performance
            self._runs += runs
            mask = (self._runs > 0)
            self._performance[mask] = self._performancesum[mask] / self._runs[mask]
            
        #degrade the exploration factor.
        self._explorationFactor *= ( 1.0 - ( self._explorationDegrationFactor / self._noOfAlgorithms ) )

