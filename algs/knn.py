'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the K-Nearest Neighbours algorithm selection method.

Principle:
It collects all the historical results. As soon as a new problem instance arrives, it uses the historical resutls to predict the highest score and runtime. Then it runs the algorithm with the best score/cost ratio that is predicted to finish withing the predicted runtime.

Drawbacks:
The more results are gathered, the slower the method gets, because all the historically results are stored. The slowness could be countered by structuring the search space, but the space requirement remains.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import distance_matrix
### custom (local) imports ### 
from .base.framework import AlgorithmSelector
from .base.framework import OutOfTimeException


class KNeighborsSelector(AlgorithmSelector):
    '''
        algorithms - vector of algorithms to run
        callback - callback function that takes: algorithm, problem -> score, cost (budget cost)
            
    '''
    def __init__(self, algorithms):
        self._algorithms = algorithms
        self._noOfAlgorithms = np.shape(self._algorithms)[0]
        self._runs = np.zeros(self._noOfAlgorithms)
        self._score_history = [] #the results of past games.
        self._cost_history = [] #the results of past games.
        self.k = None #set to None to take all samples into account (weighted by their distance). Set to a number if you want to consider only the k nearest.

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
        
        def getClosest(refValues, currentValues, finalValues):
            '''
                Take a refererence value matrix (refValues) of size n x d
                And a currentValues vector of size 1 x d
                And a finalValues matrix of size n x d
                
                Calculate the euclidean distances of the currentValues to all the refValues. (Do ignore values that are zero?)
                
                Take the inverse distances as weights to aggregate the finalValues. Ignore zeros (= that sample cannot provide any information so it should not influence the prediction)
                Return thoose values as the predicted ones.
                
                If an empty refValues matrix is provided, return a random vector of the size of the currentValues vector.
            '''
            #TODO: Norm the refValues and then also the currentValues
            #print("===")
            #print("currentValues")
            #print(currentValues)
            #Fix if currentValues is a 1-d vector instead of a 1 x d matrix:
            if len(np.shape(currentValues))==1:
                currentValues = np.reshape(currentValues, (1,np.shape(currentValues)[0]))
                
            d = np.shape(currentValues)[1]

            #if we get an empty reference matrix, we just randomly guess results:
            if len(refValues) == 0:
                return np.random.rand(d)
            
            n = np.shape(refValues)[0]
            
            #we set all the values where we do not have information about in the current value to zero in the reference matrix. Thoose will match to a distance of zero but will also be weighted accordingly.
            refValues = refValues.copy()
            refValues[:,(currentValues == 0)[0]] = 0
            
            #print("refValues (after zeroing)")
            #print(refValues)
            
            #distance matrix should have size n x 1
            distances = distance_matrix(refValues, currentValues)
            #print("distances")
            #print(distances)
            distances = distances * distances #square such that lower distances are better discriminated from higher ones.
            distances = ( distances / np.linalg.norm(distances, ord = 1) )#divide via max-norm, then inverse such that distances of zero become one and max distances become zero.
            distances[np.isnan(distances)] = 0
            weights = 1.0 - distances
            weights += regularization
            
            if not( self.k is None):
                indexes = np.argsort(weights)
                #only if there are more than 3 weights we set the rest to zero.
                if (len(indexes)> self.k):
                    weights[indexes[ self.k:]] = 0
            
            #now we need to calculate the estimate for each of the d output values.
            #weights[n,1]
            #finalValues[n,d]
            mask = (finalValues == 0)
            weights = np.repeat(weights, repeats = d, axis = 1)
            #print("weights (after repeat)")
            #print(weights)
            #set all weights where the finalValues are zero to zero.
            weights[mask] = 0
            #print("weights (after mask)")
            #print(weights)
            #normalize the weights for each d seperately so they sum up to 1:
            weights = weights / np.sum(weights, axis = 0)
            weights[np.isnan(weights)] = 0
            #calculate the weighted sums:
            #print("weights")
            #print(weights)
            #print("finalValues")
            #print(finalValues)
            predictions = np.sum(finalValues * weights, axis = 0)
            #print("predictions")
            #print(predictions)
            return predictions
 
        

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            hscores = np.array( self._score_history )
            hcosts = np.array( self._cost_history)

            scorePredictions = getClosest(hscores, scores, hscores)[algorithms]
            costPredictions = getClosest(hcosts, costs, hcosts)[algorithms]
            
            scorePredictions = scorePredictions / costPredictions
            scorePredictions[np.isnan(scorePredictions)] = 0
            #print("hscores")
            #print(hscores)
            #print("Current Scores")
            #print(scores)
            #print("Predictions")
            #print(scorePredictions)

            #regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
            #regressor.fit(hscores)

            #set thoose algorithms that are likely to not run within the remaining budget anymore to zero so they are not choosen:
            scorePredictions[ costPredictions > remainingBudget ] = 0
            
            #normalize the score predictions to sum up to 1:
            if (np.sum(scorePredictions) == 0):
                scorePredictions += regularization #to prevent complete zeros
            scorePredictions = scorePredictions / np.sum(scorePredictions)
            
            selectedAlgorithm = np.random.choice(a = algorithms, replace = False, p = scorePredictions)
            #selectedAlgorithm = algorithms[np.argmax(scorePredictions)]

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
            self._score_history.append(scores)
            unique, unique_indices = np.unique(self._score_history, return_index = True, axis = 0)
            self._score_history = unique.tolist()
            
            self._cost_history.append(costs)
            unique, unique_indices = np.unique(self._cost_history, return_index = True, axis = 0)
            self._cost_history = unique.tolist()
            
            self._runs += runs
            


