'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the nemesis method, based on an idea from Christian Geißler (2019).
It counts how often an algorithm was better than another one evaluated on the same dataset, constructing a matrix of wins/looses for each pair of algorithms.

For a new algorithm selection process, it will start with the algorithm than won the most. After evaluating it, it will select the next algorithm based on which one beated the best one evaluated on the current problem instance. We therefore confront the empirical best one with thoose that most often beat the current champion, therefore maximizing the chance of beating it again.

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


class NemesisSelector(AlgorithmSelector):
    '''
        algorithms - vector of algorithms to run
        callback - callback function that takes: algorithm, problem -> score, cost (budget cost)
            
    '''
    def __init__(self, algorithms):
        self._algorithms = algorithms
        self._noOfAlgorithms = np.shape(self._algorithms)[0]
        self._performancesum = np.zeros(self._noOfAlgorithms)
        self._runs = np.zeros(self._noOfAlgorithms)
        self._nemesismatrix = np.zeros((self._noOfAlgorithms,self._noOfAlgorithms))

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
        
        currentBestAlgorithm = None

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
        
            if ( currentBestAlgorithm is None ):
                selectedAlgorithm = np.argmax( np.sum( self._nemesismatrix, axis = 1 ) )
            else:
                selectedAlgorithm = algorithms[ np.argmax( self._nemesismatrix[algorithms,currentBestAlgorithm] ) ]
                


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
            
            #update the best algorithm
            if (currentBestAlgorithm is None):
                currentBestAlgorithm = selectedAlgorithm
            elif ( score >= np.max(scores) ):
                currentBestAlgorithm = selectedAlgorithm


        if ( np.max(scores)!=0 ) & ( np.max(costs) != 0 ):
            #update the nemesis matrix:
            
            scores = scores[runs]
            scoreMatrix = np.repeat( np.expand_dims(scores, axis = 1), len(scores), axis = 1 )
            scoreMatrix = scoreMatrix - scoreMatrix.T
            scoreMatrix[scoreMatrix > 0] = 1
            scoreMatrix[scoreMatrix < 0] = -1
            
            #now, scoreMatrix[0] contains all wins and losses from the perspective of algorithm 0.

            #np.expand_dims(runs, axis=1) * runs
            for i in range(np.shape(scoreMatrix)[0]):
                ai = (self._algorithms[runs])[i]
                for i2 in range(np.shape(scoreMatrix)[0]):
                    ai2 = (self._algorithms[runs])[i2]
                    self._nemesismatrix[ai,ai2] = self._nemesismatrix[ai,ai2] + scoreMatrix[i,i2]

