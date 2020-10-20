'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of our bayesian optimization inspired approach to discrete algorithm selection. 

The approach is as follows:

We consider two different processes, the global meta-learning process and the locla problem-instance process. The local problem-instance process deals with optimizing the area under the curve of selecting a sequence of algorithms for one single problem instance with respect to a given, problem instance specific amount of budget. The global meta-learning process takes theese local processes into account and learns to generalize the algorithm performances and their relationships.

Both processes interact with each other: The global process provides the initial priors to the local process and a way to update them when new information has been observed. The local process provides the observed runtime information back to the global process.


For each algorithm, we assume a gaussian distribution of it's performance over the different problem instances. We further assume a correlation between the performance values (their value and variance) between the different algorithms. We use theese correlations to weight the influence of linear predictors that modify the priors of our model. After this, we calculate the predicted performance values (mean and variance) and choose the most promising one with an expected improvement function.

Initially, the mean and variance of each algorithm in the set is estimated based on historical observations, which represents our posteriors. We then pick an algorithm based on the aquisition function which uses estimated improvement measures to select the most promising algorithm given the current assumptions. After we have obtained the results, we update the local model

 But while optimizing, the observations are adjusted  only based on historical occurences and a simple metric that weights score vs. time.
 
Source:
http://papers.nips.cc/paper/3219-active-preference-learning-with-discrete-choice-data.pdf

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import StandardScaler
### custom (local) imports ### 
from .base.framework import AlgorithmSelector
from .base.framework import OutOfTimeException


class BayesianOptimizationAlgorithmSelector(AlgorithmSelector):
    '''
        algorithms - vector of algorithms to run
        callback - callback function that takes: algorithm, problem -> score, cost (budget cost)
            
    '''
    def __init__(self, algorithms):
        self._algorithms = algorithms
        self._noOfAlgorithms = np.shape(self._algorithms)[0]
        
        #self._performancesum = np.zeros(self._noOfAlgorithms)
        self._runs = np.zeros(self._noOfAlgorithms)
        
        #normalized scores and costs
        self.history_fillPointer = 0
        self.history_incrementSize = 1000
        self.history_shape = (self._noOfAlgorithms, self.history_incrementSize)
        #self._normalized_scores = csc_matrix(history_shape, dtype='d')
        #self._normalized_costs = csc_matrix(history_shape, dtype='d')
        self._normalized_performances = csc_matrix(self.history_shape, dtype='d')
        
        #derived properties:
        self._mean_normalized_performances = np.zeros(self._noOfAlgorithms)
        self._variance_normalized_performances = np.zeros(self._noOfAlgorithms)
        self._performance_correlations = np.zeros((self._noOfAlgorithms, self._noOfAlgorithms))
        
        #fix set the internal hyperparameters. They are quite robust and only change the behaviour if set to extreme values.
        self._explorationFactor = 1.0 #how much weight we put in exploration
        self._explorationDegrationFactor = 1.0 #hyperparameter between 0 and 1
    
    def compute_sparse_correlation_matrix(self, A):
        '''
            SOURCE of this function: https://stackoverflow.com/a/59532845/1734298
        '''
        scaler = StandardScaler(with_mean=False)
        scaled_A = scaler.fit_transform(A)  # Assuming A is a CSR or CSC matrix
        corr_matrix = (1/scaled_A.shape[0]) * (scaled_A.T @ scaled_A)
        return corr_matrix
    
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
            remainingPerformances = self._mean_normalized_performances[algorithms]
            
            correlationMatrix = self._variance_normalized_performances.copy()
            
            performanceEstimation = self._mean_normalized_performances
            
            nscores = scores / np.max(scores)
            ncosts = costs / np.max(costs)
            #overwrite normalized mean values with normalized true observed values
            performanceEstimation[runs] = (nscores[runs]).flatten() / (ncosts[runs]).flatten()
            
            #norm matrix such that each column sums up to one:
            correlationMatrix = correlationMatrix / np.sum(correlationMatrix, axis=0)
            #then multiply the roughly estimated performance estimations with the correlation matrix (we basically weight each est. performance value by the correlation)
            performanceEstimation = performanceEstimation * correlationMatrix
            #and now we keep only thoose estimated performance values that remain to be evaluated:
            remainingPerformances = performanceEstimation[algorithms]
            
            self._performance_correlations[algorithms,:]
            #Choose the algorithm randomly with a probability proportionally to it's historical performance
            #print("no of dimensions: "+str(np.shape(remainingPerformances)))
            

            probabilities = remainingPerformances + self._explorationFactor * ( (np.sum(remainingPerformances)==0) + np.linalg.norm(remainingPerformances, ord=1) / len(remainingPerformances) )
            
            normOfProbabilities = np.linalg.norm(probabilities, ord=1)
            if ( normOfProbabilities > 0 ):
                probabilities = probabilities / normOfProbabilities
            else:
                probabilities[:] = 1

            probabilities = (np.array(probabilities)).flatten()
            
            probabilities = probabilities / np.sum(probabilities)

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
            
            #calculate and add the normalized performance to the history:
            performance = scores[runs] / costs[runs]
            #print('performance')
            #print(performance)
            self._normalized_performances[runs, self.history_fillPointer] = performance
            self._runs += runs
            
            submatrix = self._normalized_performances[:, :self.history_fillPointer+1]
            #print('submatrix')
            #print(submatrix)
            if ( np.sum(submatrix) <= 0 ):
                self._mean_normalized_performances = np.ones((np.shape(submatrix)[0],1))
            else:
                filledNormalizedPerformances = self._normalized_performances[:, :self.history_fillPointer+1]
                print(np.shape(filledNormalizedPerformances))
                self._mean_normalized_performances = np.mean(filledNormalizedPerformances, axis = 1)
                if self.history_fillPointer > 1:
                    self._variance_normalized_performances = np.var(filledNormalizedPerformances, axis = 1)
                else:
                    self._variance_normalized_performances[:] = 0
            
            #update correlation matrix
            self._performance_correlations = self.compute_sparse_correlation_matrix(self._normalized_performances[:,:self.history_fillPointer+1])
            
            
            self._performance_correlations = self._performance_correlations.todense()
            #print('_performance_correlations')
            #print(np.shape(self._performance_correlations))
            #if (np.sum(np.shape(self._performance_correlations))>2):
            #    print(np.diagonal(self._performance_correlations))
            
            self.history_fillPointer += 1
            
            if ( self.history_fillPointer > np.shape(self._normalized_performances)[1] ):
                self._normalized_performances = self._normalized_performances, csc_matrix(self.history_shape, dytype='d')
            
            
        #degrade the exploration factor.
        self._explorationFactor *= ( 1.0 - ( self._explorationDegrationFactor / self._noOfAlgorithms ) )

