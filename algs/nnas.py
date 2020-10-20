'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the Neural Network Algorithm Selector method, based on learning from historical data.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 
import numpy as np
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
### custom (local) imports ### 
from .base.framework import AlgorithmSelector
from .base.framework import OutOfTimeException


class NNModelWrapper():
    baseNetwork = None
    
    def __init__(self, noOfAlgorithms):
        
        net = nn.Sequential()
        net.add(
            self.createBaseNetwork(noOfAlgorithms),
            nn.Dense(1, flatten=True),
        )
        net.initialize(init=init.Xavier(), force_reinit = False) #force_reinit 
        self.model = net
        self.objectiveFunction = gluon.loss.L2Loss()
        self.trainer = gluon.Trainer(net.collect_params(), 'adagrad', {'learning_rate': 0.1})
        self.accLoss = 0
        
    def createBaseNetwork(self, noOfAlgorithms):
        if (NNModelWrapper.baseNetwork is None):
            net = nn.Sequential()
            net.add(
                nn.Dense(2*noOfAlgorithms, activation = "relu"),
                nn.Dense(2*noOfAlgorithms, activation = "relu"),
                #nn.Dense(4*noOfAlgorithms, activation = "relu"),
                #nn.Dense(3*noOfAlgorithms, activation = "relu"),
                #nn.Dense(2*noOfAlgorithms, activation = "relu"),
                #nn.Dense(1*noOfAlgorithms, activation = "relu"),
                #nn.Dense(1, flatten=True),
                #nn.Dense(1*noOfAlgorithms, activation = "relu"),
                #nn.GlobalAvgPool1D(),
                #nn.Flatten()
            )
            #nn.initialize(init=init.Xavier())
            NNModelWrapper.baseNetwork = net
        return NNModelWrapper.baseNetwork
        
    def partial_fit(self, X, y):
        X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        y = nd.array(y)
        with autograd.record():
            output = self.model(X)
            loss = self.objectiveFunction(output, y)
        loss.backward()
        self.trainer.step(np.shape(y)[0])
        
    def predict(self, X):
        X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        y = (self.model(X).asnumpy())[0]
        return y
        
    def score(self, X, y):
        X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        y = nd.array(y)
        output = self.model(X)
        return self.objectiveFunction(output, y)
            
'''
    Basic approach:
    
    When starting a new process, we update our model with the past data.

'''
class NeuralNetworkAlgorithmSelector(AlgorithmSelector):
            
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
        self._explorationFactor = 0.0 #how much weight we put in exploration
        self._explorationDegrationFactor = 0.0 #hyperparameter between 0 and 1
        self._initialCollectionThreshold = 1
        
        #initialize a score model for each algorithm
        #model = PassiveAggressiveRegressor
        #model = SGDRegressor
        
        self._scoreModels = np.array([NNModelWrapper(noOfAlgorithms = self._noOfAlgorithms) for a in range(self._noOfAlgorithms)])
        self._costModels = np.array([NNModelWrapper(noOfAlgorithms = self._noOfAlgorithms) for a in range(self._noOfAlgorithms)])
        self._lowestCost = None
        self._debugDisplayCounter = 0
        

    def updateModels(self, scores, costs, runs):
        # permute the training set #
        #first, count how many samples we have in the dataset:
        algorithmIndexes = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        
        algorithmIndexes = algorithmIndexes[runs[:,0]>0]
        noOfRuns = np.sum(runs >= 0)
        
        cost_state = np.concatenate((costs, runs), axis=0)
        score_state = np.concatenate((scores, runs), axis=0)
        '''
        print(' ')
        print('=== updateModels ===')
        print('costs')
        print(costs)
        print('runs')
        print(runs)
        print('algorithm indexes')
        print(algorithmIndexes)
        '''
        
        offset = len(scores)
        for i in algorithmIndexes:
            tmp_score = scores.copy()
            tmp_costs = costs.copy()
            tmp_runs = runs.copy()
            #remove target values from the features:
            tmp_score[i] = 0
            tmp_costs[i] = 0
            tmp_runs[i] = 0 
            np.concatenate((tmp_costs, tmp_runs), out=cost_state, axis=0)
            np.concatenate((tmp_score, tmp_runs), out=score_state, axis=0)
            self._costModels[i].partial_fit(cost_state.T, [costs[i]])
            self._scoreModels[i].partial_fit(score_state.T, [scores[i]])


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
        cost_state = np.concatenate((costs, runs), axis=0)
        score_state = np.concatenate((scores, runs), axis=0)

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            if ( np.any( self._runs < self._initialCollectionThreshold ) ):
                estimatedCosts = np.ones(len(algorithms))
                estimatedScores = np.ones(len(algorithms))
            else:
                estimatedCosts = np.array([estimator.predict(cost_state.T) for estimator in costEstimators])[:,0]
                estimatedScores = np.array([estimator.predict(score_state.T) for estimator in scoreEstimators])[:,0]
                
            #make sure there are no negative values:
            estimatedCosts[estimatedCosts <= 0] = 1
            estimatedScores[estimatedScores < 0] = 0


            #Old score/costs strategy:
            estimatedPerformances = estimatedScores / estimatedCosts
            
            #New strategy
            
            #estimatedPerformances = estimatedScores - np.min(estimatedScores)
            #bestIndex = np.argmax(estimatedScores)
            #estimatedPerformances[:] = 0
            #estimatedPerformances[bestIndex] = estimatedPerformances[bestIndex] * 2.0 #boost the score of the best one

            #we add a little curiosity to improve the background model.
            weights = estimatedPerformances + ( self._explorationFactor * np.linalg.norm(estimatedPerformances) )
            
            #we are not considering thoose where we think we exceed the remaining Budget anyways.
            weights[estimatedCosts > remainingBudget] = 0
            
            
            if ( np.all(weights==0) ):
                weights[:] = 1.0
            #Choose the algorithm randomly with a probability proportionally to it's historical performance
            weights = np.nan_to_num(weights, 0.0)
            if (np.sum(weights) == 0):
                weights[:] = 1
            weights = np.power(weights,2)
            probabilities = weights / np.linalg.norm(weights, ord=1)
            selectedAlgorithm = np.random.choice(a = algorithms, replace = False, p = probabilities)
            #selectedAlgorithm = algorithms[np.argmax(probabilities)]
            
            #run the algorithm and get the budget and cost back
            self._runs[selectedAlgorithm] += 1
            outOfBudget = False
            try:
                ( cost, score ) = callback( self._algorithms[ selectedAlgorithm ])
                
                #cdiff = np.abs( cost - estimatedCosts[algorithms == selectedAlgorithm] )
                #sdiff = np.abs( score - estimatedScores[algorithms == selectedAlgorithm] )
                #print("cdiff: "+str(cdiff)+" sdiff:"+str(sdiff))
            except OutOfTimeException:
                cost = remainingBudget
                score = 0
                outOfBudget = True
            
            #if we exceeded the budget, this run does not count.
            if (remainingBudget < cost ):
                cost = remainingBudget
                score = 0
                outOfBudget = True
                
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
            
            

            if ( outOfBudget == False ):
                costs[selectedAlgorithm] = cost
                scores[selectedAlgorithm] = score
                runs[selectedAlgorithm] = True
                np.concatenate((costs, runs), out=cost_state, axis=0)
                np.concatenate((scores, runs), out=score_state, axis=0)
                self.updateModels(scores.copy(), costs.copy(), runs.copy())
            
        # DEBUGGING: Test prediction quality:
        '''
        if ( self._debugDisplayCounter % 100 == 0 ):
            if ( np.any( self._runs >= self._initialCollectionThreshold ) ):
                estimatedCosts = np.array([estimator.predict(cost_state.T) for estimator in self._costModels])[:,0]
                estimatedScores = np.array([estimator.predict(score_state.T) for estimator in self._scoreModels])[:,0]
                mask = (runs[:,0]>0)
                print("Summary")
                #print("cost state")
                #print(str(cost_state.T))
                print("Measured vs estimated costs")
                print(costs[mask,0])
                print(estimatedCosts[mask])
                #print("score state")
                #print(str(score_state.T))
                print("Measured vs estimated scores")
                print(scores[mask,0])
                print(estimatedScores[mask])
                print("runs")
                print(self._runs)
                print("exploration factor: "+str(self._explorationFactor))
        self._debugDisplayCounter += 1
        '''
        
            
        #degrade the exploration factor.
        self._explorationFactor *= ( 1.0 - ( self._explorationDegrationFactor / self._noOfAlgorithms ) )

