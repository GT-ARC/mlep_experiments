'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the average ranking method, based on (Lin, S. (2010). Rank aggregation methods. WIREs Computational Statistics, 2, 555–570. )
but slightly adjusted. It chooses the algorithm that performed best on average in the past first, then second best etc.
Instead of averaging over the raw algorithm score, we average over the min/max normalized score of each algorithm selection process.
Note: Does not take runtime into account.


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


class AverageRankingSelector(AlgorithmSelector):
    '''
        algorithms - vector of algorithms to run
        callback - callback function that takes: algorithm, problem -> score, cost (budget cost)
            
    '''
    def __init__(self, algorithms):
        self._algorithms = algorithms
        self._noOfAlgorithms = np.shape(self._algorithms)[0]
        self._performancesum = np.zeros(self._noOfAlgorithms)
        self._runs = np.zeros(self._noOfAlgorithms)

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
        self._performance = np.zeros(self._noOfAlgorithms)
        remainingBudget = budget

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            remainingPerformances = self._performance[algorithms]
            

            #Choose the best remaining algorithm
            selectedAlgorithm = algorithms[ np.argmax(remainingPerformances) ]

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
            

        if ( np.max(scores)!=0 ) & ( np.max(costs) != 0 ):
        
            def minMaxNorm(s):
                minValue = np.min( s )
                maxValue = np.max( s )
                diff = maxValue - minValue
                if (diff == 0):
                    s[:] = 0.5
                else:
                    s = ( ( s - minValue ) / diff )
                return s
            
            performance = minMaxNorm(scores[runs])
            self._performancesum[runs] += performance
            self._runs += runs
            mask = (self._runs > 0)
            self._performance[mask] = self._performancesum[mask] / self._runs[mask]

