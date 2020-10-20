"""
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains helper functions to evaluate algorithm selection implementations.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
"""

### python basic imports ###
import time, datetime
from pathlib import Path
import json
### 3rd party imports (from packages, the environment) ### 
import numpy as np
### custom (local) imports ### 
from .framework import OutOfTimeException

'''    

'''
class AlgorithmSelectionProcess:

    """
    A helper class for evaluation algorithm selection processes.
    It wraps around the callback function and keeps track of results and budgets.
    On default, it also measures the time a callback requires.
    For the evaluation, it can calculate area under the curve metrics over the recorded performance values to calculate the anytime performance metric.
    
    Intended usage
    --------------
    Create an instance of this if you start to evaluate different algorithm calls. Then make the calls via this objects callback function. After or during the calls, use the methods to display the progress or final result.
    """
    
    def __init__(self, budget, callback, measuretime = True):
        """
        Constructor that initializes a new algorithm selection process.
        
        Parameters
        ----------
        budget : float
            time budget for the whole process in seconds
        callback : FUNCTION
            callback function that takes the algorithm as argument and returns the score the algorithm achieved
            if measuretime == False then the callback function is expected to return a tuple: (duration, score)
            with duration : int the time in seconds it took to run the callback function and score : float the score to be maximized.
            this is for supporting artificial benchmarks where the callback function is simulated instead of actually run.
        measuretime : boolean
            defaults to True, determines if this class measures the runtime of the callback function or not. See callback : FUNCTION parameter for additional effects.
            
        """
        self._budget = budget
        self._remainingBudget = budget
        
        if ( measuretime ):
            def _callbackWithTimer(algorithm):
                starttime = time.clock()
                score = callback(algorithm)
                endtime = time.clock()
                duration = endtime - starttime
                return ( duration, score )
        
            self._callback = _callbackWithTimer     
        else:
            self._callback = callback
        
        #for benchmarking
        self._scores = list()
        self._budgetspend = list()
        self._selectedAlgorithm = list()
        
        self._bestscores = list()
        self._leftbudget = list()
        self._bestalgorithms = list()
        
    
    def callback(self, algorithm):
        """
        Mostly internally used function that wraps around calling the actual internal callback function provided at construction time.
        
        Parameters
        ----------
        algorithm : same type as the initially provided callback function.
        
        Returns
        -------
        tuple (duration:int, score:float)
            with duration : int the time in seconds it took to run the callback function and score : float the score to be maximized.
        
        
        Exceptions
        ----------
        OutOfTimeException : Exception
            when the callback function exceeded the remaining time budget.
            
        """
        if ( self._remainingBudget > 0 ):
            ( duration, score ) = self._callback(algorithm)
            self._remainingBudget -= duration
            
            #store the results for later score calculation
            if ( self._remainingBudget > 0 ):
                #store the score
                self._scores.append(score)
                self._budgetspend.append( self._budget - self._remainingBudget)
                self._selectedAlgorithm.append(algorithm)
                #store best scores if they are better.
                if ( ( not self._bestscores ) or ( score > self._bestscores[-1] ) ):
                    self._bestscores.append(score)
                    self._leftbudget.append(self._remainingBudget)
                    self._bestalgorithms.append(algorithm)
                #return the result
                return ( duration, score )
        raise OutOfTimeException("The time budget for the algorithm has been exceeded")
        
    def absAUC(self):
        """
        Returns
        -------
        float
            the absolute score (area under the score/time curve).
        """
        areaUnderScore = 0
        nextBudget = 0
        if ( not self._bestscores ):
            return 0
        #traverse the reversed list:
        for (score, leftbudget) in reversed( list( zip(self._bestscores, self._leftbudget ) ) ):
            areaUnderScore += score * np.abs( nextBudget - leftbudget ) #difference to the next best score update
            nextBudget = leftbudget
        return areaUnderScore
        
    

    def normalizedAUC(self, maxAUCScore, minAUCScore = 0):
        """
        Returns
        -------
        float
            the area under the curve is normalized to strip it from differences coming from:
                different problem instance budgets (by dividing by the problem instance budget)
                different problem difficulties (via max/min normalization)
            the normalized score (area under the score/time curve). A value between 0 and 1, where 1 is the best.
        """
        return ( (self.absAUC() - minAUCScore ) / ( maxAUCScore - minAUCScore  ))
            
    def getCompleteProcessRecord(self):
        """
        Method that returns a list of all the observed objective function calls and their respective parameters and recorded features.
        
        Returns
        -------
        list
            a list with each object being a triple of (budgetspend:float, score:float, algorithm:any)
        """
        if ( not self._scores ):
            return (None, None, None)
        return list(zip(self._budgetspend, self._scores, self._selectedAlgorithm))
        
    def getBest(self):
        """
        Get the best score and algoritm seen so far during this evaluation.
        Returns
        -------
        tuple(score : float, algorithm : any)
        """
        if ( not self._bestscores ):
            return (None, None)
        return (self._bestscores[-1], self._bestalgorithms[-1])
        
class AlgorithmSelectionProcessRecorder:

    """
    A helper class to save the algorithm selection process results to a file to be plotted in the future.
    """
    
    def __init__(self, experiment_id, noOfAlgorithms, noOfProblems, namesOfAlgorithms ):
        self.experiment_id = experiment_id
        self.shapeOfResults = (noOfAlgorithms, noOfProblems)
        self.metadata = dict()
        self.results = dict()
        self.metadata['experiment_id'] = experiment_id
        self.metadata['noOfAlgorithms'] = noOfAlgorithms
        self.metadata['noOfProblems'] = noOfProblems
        self.metadata['namesOfAlgorithms'] = namesOfAlgorithms
        self.metadata['datetime[utc]'] = str(datetime.datetime.now(datetime.timezone.utc))
        
    def setMetadata(self, key, value):
        self.metadata[key] = value
        
    def addRecord(self,algorithmIndex, problemIndex, resultDict):
        for key, value in resultDict.items():
            if not (key in self.results):
                self.results[key] = ( np.zeros(self.shapeOfResults) ).tolist()
            self.results[key][algorithmIndex][problemIndex] = value
            
    def saveRecords(self, filedir):
        Path(filedir).mkdir(parents=True, exist_ok=True)
        
        filenameExists = True
        counter = 0
        while(filenameExists):
            filename = self.experiment_id + '_' + str(counter)
            filepath = Path(filedir + filename + '.json')
            filenameExists = filepath.exists()
            counter += 1
            
        with open(filepath, 'w') as outfile:
            json.dump([self.metadata,self.results], outfile)
        
def calculateAUCViaCallback(sequenceOfElements, callback, budget):
    remainingBudget = budget
    areaUnderScore = 0
    bestScore = 0
    for e in sequenceOfElements:
        (cost, score) = callback(e)
        areaUnderScore += min(cost, remainingBudget) * bestScore
        remainingBudget = remainingBudget - cost
        if ( remainingBudget >= 0 ):
            if (bestScore < score):
                bestScore = score
        else:
            break
    #if there is still budget left, add the payout over that remaining budget to the aus.
    areaUnderScore += max( remainingBudget, 0 ) * bestScore
    return areaUnderScore  
       
def calculateAUC(sequenceOfElements, costs, scores, budget):
    remainingBudget = budget
    areaUnderScore = 0
    bestScore = 0
    
    for e in sequenceOfElements:
        cost = costs[e]
        areaUnderScore += min(cost, remainingBudget) * bestScore
        remainingBudget = remainingBudget - cost
        if ( remainingBudget >= 0 ):
            score = scores[e]
            if (bestScore < score):
                bestScore = score
        else:
            break
    #if there is still budget left, add the payout over that remaining budget to the aus.
    areaUnderScore += max( remainingBudget, 0 ) * bestScore
    return areaUnderScore     
        
def getOptimalSequence(elements, o_costs, o_scores, budget):
    '''
    Calculating the optimal schedule.
    Authors: Vincent Froese & Christian Geißler
    (c) 2020
    '''
    #keep only elements with cost lower than the originalBudget
    indexes = (o_costs[elements] < budget)
    elements = elements[indexes].copy()
    costs = o_costs[indexes].copy()
    scores = o_scores[indexes].copy()
    #sort the elements according to an ascending score:
    ascendingSortedIndexes = np.argsort(scores)
    elements = elements[ascendingSortedIndexes]
    costs = costs[ascendingSortedIndexes]
    scores = scores[ascendingSortedIndexes]
    
    kMax = len(elements)
    
    bestValueForAnyK = 0
    bestSequenceForAnyK = []

    #calculate B', the hypothetical budget required to execute all elements.
    budgetStripe = np.sum(costs)
    
    for k in range(kMax):
        #print("k = "+str(k))
        #elements = elements[:k]
        #costs = costs[:k]
        #scores = scores[:k]
        
        #A[i] = max auc of schedule that starts with job i and ends with job k with budget B'
        A = np.zeros(shape = k+1)
        ASeq = [[] for i in range(k+1)]
        #A[k] == area of schedule that starts with k and ends with k.
        A[k] = (budgetStripe - costs[k]) * scores[k]
        ASeq[k] = [elements[k]]
        
        for i in range(k-1, -1, -1):
            for I in range( i+1, k+1, 1):
                candidate = A[I] + costs[I] * scores[i] - costs[i] * scores[k]
                if (candidate > A[i]):
                    A[i] = candidate
                    ASeq[i] = [elements[i]] + ASeq[I]

        bestA = np.argmax(A)
        bestAUCBStripe = A[bestA]
        bestSequenceForK = ASeq[bestA]
        bestAUC = bestAUCBStripe - (budgetStripe - budget) * scores[k]
        #bestAUCBStripe = calculateAUC(bestSequenceForK, o_costs, o_scores, budgetStripe)
        #bestAUC = calculateAUC(bestSequenceForK, o_costs, o_scores, budget)
        
        #print("A: "+str(A))
        #print("A[bestA]"+str(bestValueForK))
        #print("bestAUCBStripe: "+str(bestAUCBStripe))
        #print("bestAUC: "+str(bestAUC))
        if bestAUC > bestValueForAnyK:
            bestValueForAnyK = bestAUC
            bestSequenceForAnyK = bestSequenceForK
    #print("bestValueForAnyK: "+str(bestValueForAnyK))
    #print("bestSequenceForAnyK: "+str(bestSequenceForAnyK))
    return bestSequenceForAnyK
    
def oracle(algorithms, callback, budget):
    bScore = 0
    bCost = 0
    bAlgorithm = -1
    
    nAlgs = len(algorithms)
    algIndexes = np.arange(nAlgs)
    costs = np.zeros(shape = nAlgs)
    scores = np.zeros(shape = nAlgs)
    
    for ai in algIndexes:
        a = algorithms[ai]
        (cost, score) = callback(a)
        costs[ai] = cost
        scores[ai] = score
        if ( cost <= budget ) & ( score > bScore ):
            bScore = score
            bCost = cost
            bAlgorithm = a
            
    bSequence = getOptimalSequence(algorithms, costs, scores, budget)
    bAUC = calculateAUCViaCallback(bSequence, callback, budget)
    return (bCost, bScore, bAlgorithm, bSequence, bAUC)