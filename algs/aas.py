'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the advanced algorithm selection method, based on learning from historical data.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 
import numpy as np
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
### custom (local) imports ### 
from .base.framework import AlgorithmSelector
from .base.framework import OutOfTimeException

'''
    Basic approach:
    
    When starting a new process, we update our model with the past data.

'''
class AdvancedAlgorithmSelector(AlgorithmSelector):
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
        self._explorationDegrationFactor = 0.0 #hyperparameter between 0 and 1
        self._initialCollectionThreshold = 10
        
        #initialize a score model for each algorithm
        class LogScaleWrappedRegressor():
            def __init__(self, regressorInstance):
                self.regressor = regressorInstance
            
            def partial_fit(self, X, y):
                self.regressor.partial_fit(X, np.log(y+1))
            
            def predict(self, X):
                return np.exp(self.regressor.predict(X))-1
        
        self._scoreModels = np.array([SGDRegressor(penalty='l1', max_iter=1) for a in range(self._noOfAlgorithms)])
        self._costModels  = np.array([SGDRegressor(penalty='l1', max_iter=1) for a in range(self._noOfAlgorithms)])
        #self._costModels = np.array([LogScaleWrappedRegressor(SGDRegressor(penalty='l1', max_iter=1)) for a in range(self._noOfAlgorithms)])
        self._lowestCost = None
        
    def updateModels(self, scores, costs, runs):
        # permute the training set #
        #first, count how many samples we have in the dataset:
        algorithmIndexes = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        
        algorithmIndexes = algorithmIndexes[runs[:,0]>0]
        noOfRuns = np.sum(runs >= 0)
        
        state = np.concatenate((scores, costs, runs), axis=0)
        yCost  = np.zeros(noOfRuns)
        yScore  = np.zeros(noOfRuns)
        offset = len(scores)
        for i in algorithmIndexes:
            np.concatenate((scores, costs, runs), out=state, axis=0)
            state[i] = 0
            state[i+offset] = 0
            state[i+2*offset] = 0  
            self._costModels[i].partial_fit(state.T, np.ravel([costs[i]]))
            self._scoreModels[i].partial_fit(state.T, np.ravel([scores[i]]))


    '''
        run the algorithm selection on one problem instance
    '''
    def process(self, budget, callback):
        #print("past performances: "+str(self._performance))
        #print("calls: "+str(self._runs))
        #set of the indexes of the remaining algorithms
        algorithms = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        costEstimators = self._costModels[algorithms]
        scoreEstimators = self._scoreModels[algorithms]
        remainingBudget = budget
        
        #STATE
        scores = np.zeros((np.shape(self._algorithms)[0],1))
        costs = np.zeros((np.shape(self._algorithms)[0],1))
        runs = np.zeros((np.shape(self._algorithms)[0],1), dtype=bool)
        state = np.concatenate((scores, costs, runs), axis=0)

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            if ( np.any( self._runs < self._initialCollectionThreshold ) ):
                estimatedCosts = np.ones(len(algorithms))
                estimatedScores = np.ones(len(algorithms))
            else:
                estimatedCosts = np.array([estimator.predict(state.T) for estimator in costEstimators])[:,0]
                estimatedScores = np.array([estimator.predict(state.T) for estimator in scoreEstimators])[:,0]
                
            #make sure there are no negative values:
            estimatedCosts[estimatedCosts <= 0] = 1
            estimatedScores[estimatedScores < 0] = 0


            
            #Old score/costs strategy:
            #estimatedPerformances = estimatedScores / estimatedCosts
            
            #New strategy
            estimatedScores[estimatedCosts > remainingBudget] = 0
            estimatedPerformances = estimatedScores - np.min(estimatedScores)
            #bestIndex = np.argmax(estimatedScores)
            #estimatedPerformances[:] = 0
            #estimatedPerformances[bestIndex] = estimatedPerformances[bestIndex] * 2.0 #boost the score of the best one

            weights = estimatedPerformances + ( self._explorationFactor * np.linalg.norm(estimatedPerformances) )
            
            if ( np.all(weights==0) ):
                weights[:] = 1.0
            #Choose the algorithm randomly with a probability proportionally to it's historical performance
            probabilities = weights / np.linalg.norm(weights, ord=1)

            selectedAlgorithm = np.random.choice(a = algorithms, replace = False, p = probabilities)

            #run the algorithm and get the budget and cost back
            self._runs[selectedAlgorithm] += 1
            try:
                ( cost, score ) = callback( self._algorithms[ selectedAlgorithm ])
            except OutOfTimeException:
                cost = remainingBudget
                score = 0
            
            #if we exceeded the budget, this run does not count.
            if (remainingBudget < cost ):
                cost = remainingBudget
                score = 0
                
            #update the estimator of a specific algorithm
            # print(state.T)
            # print([cost])
            # print([score])
            # print(np.shape(state))
            
                
            #remove the selected algorithm from the local set, as well as from the estimator sets
            costEstimators = costEstimators[algorithms != selectedAlgorithm]
            scoreEstimators = scoreEstimators[algorithms != selectedAlgorithm]
            algorithms = algorithms[algorithms != selectedAlgorithm]
                
            remainingBudget -= cost
            #print("algorithm:"+str(self._algorithms[ selectedAlgorithm ])+" score: "+str(score)+" cost: "+str(cost))
            
            
            costs[selectedAlgorithm] = cost
            scores[selectedAlgorithm] = score
            runs[selectedAlgorithm] = True
            np.concatenate((scores, costs, runs), out=state, axis=0)
            
            self.updateModels(scores, costs, runs)
            
        # DEBUGGING: Test prediction quality:
        '''
        if ( np.any( self._runs >= self._initialCollectionThreshold ) ):
            estimatedCosts = np.array([estimator.predict(state.T) for estimator in self._costModels])[:,0]
            estimatedScores = np.array([estimator.predict(state.T) for estimator in self._scoreModels])[:,0]
            mask = (runs[:,0]>0)
            print("Summary")
            print("Measured vs estimated costs")
            print(costs[mask,0])
            print(estimatedCosts[mask])
            print("Measured vs estimated scores")
            print(scores[mask,0])
            print(estimatedScores[mask])
            print(self._runs[mask])
            print("exploration factor: "+str(self._explorationFactor))
        '''
        
            
        #degrade the exploration factor.
        self._explorationFactor *= ( 1.0 - ( self._explorationDegrationFactor / self._noOfAlgorithms ) )

