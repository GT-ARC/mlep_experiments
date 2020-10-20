'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the Neural Network Algorithm Selector method, based on learning from historical data.
Its a small improvement over the nnas.py version with the adjustment of using only a single neural network model with multinomial regression loss.

Since for a neural network, we need a lot of training examples, we sample them from all possible permutations after observing and deciding on a set of examples.
But instead of training the network with just that set of examples, we store them away in an example repository and draw from that repository to get training examples.

Step 1: Do the algorithm selection on one problem instance with the old models.
Step 2: After finishing, we take all the observations and generate training examples. We put them in a training example repository.
Step 3: We draw randomly from the training example repository and update the models with thoose.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 
import numpy as np
import pandas as pd
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
### custom (local) imports ### 
from .base.framework import AlgorithmSelector
from .base.framework import OutOfTimeException


class NNModelWrapperR2():
    baseNetwork = None
    
    def __init__(self, noOfAlgorithms):
        
        net = nn.Sequential()
        net.add(
            self.createBaseNetwork(noOfAlgorithms),
            nn.Dense(noOfAlgorithms, flatten=True),
        )
        net.initialize(init=init.Xavier(), force_reinit = False) #force_reinit 
        self.model = net
        self.objectiveFunction = gluon.loss.L2Loss(batch_axis = 1)
        self.trainer = gluon.Trainer(net.collect_params(), 'adam')
        #self.trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
        self.accLoss = 0
        
    def createBaseNetwork(self, noOfAlgorithms):
        if (NNModelWrapperR2.baseNetwork is None):
            net = nn.Sequential()
            net.add(
            
                #nn.Dense(8*noOfAlgorithms, activation = "relu"),
                #nn.Dense(4*noOfAlgorithms, activation = "relu"),
                #nn.Dense(2*noOfAlgorithms, activation = "relu"),
                #nn.Dense(1*noOfAlgorithms, activation = "relu"),
                
                nn.Dense(3*noOfAlgorithms, activation = "relu"),
                nn.Dense(3, activation = "relu"),
                nn.Dense(noOfAlgorithms, activation = "relu"),
                
                #nn.Dense(3*noOfAlgorithms, activation = "relu"),
                #nn.Dense(2*noOfAlgorithms, activation = "relu"),
                #nn.Dense(1*noOfAlgorithms, activation = "relu"),
                #nn.Dense(1, flatten=True),
                #nn.Dense(1*noOfAlgorithms, activation = "relu"),
                #nn.GlobalAvgPool1D(),
                #nn.Flatten()
            )
            #nn.initialize(init=init.Xavier())
            NNModelWrapperR2.baseNetwork = net
        return NNModelWrapperR2.baseNetwork
    
    
    def partial_fit(self, X, ytrue, positions):
        if len(positions) == 0:
            return
        '''
        We only use some y labels and set the loss for all the others to zero.
        '''
        X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        
        y = self.model(X)[0,:] #we set y to the current predictions of the model.
        y[positions] = ytrue[:,0]
        
        with autograd.record():
            output = self.model(X)
            loss = self.objectiveFunction(output, y)
        #print(np.shape(loss))
        #loss[positions] = 0
        #print("loss[")
        #print(loss)
        #print("]")
        loss.backward()
        self.trainer.step(np.shape(y)[0])
        
    def predict(self, X):
        X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        y = self.model(X).asnumpy()
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
DEBUG_SHOW_PREDICTION_QUALITY = False
class NeuralNetworkAlgorithmSelectorR2(AlgorithmSelector):
            
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
        self._explorationFactor = 1.0 #how much weight we put in exploration at the beginning
        self._explorationDegrationFactor = 0.01 #hyperparameter between 0 and 1
        self._initialCollectionThreshold = 1
        
        #initialize a score model for each algorithm
        #model = PassiveAggressiveRegressor
        #model = SGDRegressor
        
        self._scoreModel = NNModelWrapperR2(noOfAlgorithms = self._noOfAlgorithms) 
        self._costModel = NNModelWrapperR2(noOfAlgorithms = self._noOfAlgorithms)
        self._lowestCost = None
        self._debugDisplayCounter = 0
        
        self._trainingExamples = list()

    
    def updateModels(self, scores, costs, runs):
        #Updating the models is a two-fold step.
        #First, we generate examples from the latest algorithm selection task. We add thoose to a storage, because if we would just update the model with the current examples, we would totally overfit.
        #In the second step, we draw a certain fraction of the stored examples and use them for updating the model.
    
        # permute the training set #
        #first, count how many samples we have in the dataset:
        algorithmIndexes = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        
        algorithmIndexes = algorithmIndexes[runs[:,0]>0]
        
        #init cost and score state (actually, we only need the correct size, the values within don't matter, they will be overwritten later)
        #cost_state = np.concatenate((costs, runs), axis=0)
        #score_state = np.concatenate((scores, runs), axis=0)
        state = np.concatenate((scores, costs, runs), axis=0)
        
        tmp_score = scores.copy()
        tmp_costs = costs.copy()
        tmp_runs = runs.copy()
        
        noOfDesiredTrainingSteps = 100
        maxExamplesToKeep = noOfDesiredTrainingSteps * self._noOfAlgorithms
        
        currentNoOfExamples = len(self._trainingExamples)
        examplesCollected = 0
        noOfExamplesToCollectThisStep = ( maxExamplesToKeep + noOfDesiredTrainingSteps ) - currentNoOfExamples

        def allSubsets(elements):
            '''
                yields all possible subsets of elements without repetition
            '''
            noOfElements = len(elements)
            
            #first: return the empty set
            yield []
            
            for i in range(noOfElements):
                if noOfElements > 1:
                    for subset in allSubsets(elements[(i+1):]):
                        yield [elements[i]] + subset
                else:
                    yield [elements[i]]
                    
        def allSubsetsOfMaxLength(elements, maxlength):
            '''
            returns all subsets of elements with a maximum length of maxlength, starting with the smallest subsets.
            '''
            for i in range(maxlength+1):
                for subset in allSubsetsOfLength(elements, length = i):
                    yield subset

        def allSubsetsOfLength(elements, length):
            if length == 0:
                yield []
            elif length == 1:
                for i in elements:
                    yield [i]
            elif length > 1:
                for i in range(len(elements)):
                    for subset in allSubsetsOfLength(elements[i+1:], length = length - 1):
                        yield [elements[i]] + subset

        maxNumberOfCombinations = 0
        for p in allSubsetsOfMaxLength(algorithmIndexes, len(algorithmIndexes)):
            maxNumberOfCombinations += 1
        
        probabilityForDrawing = noOfExamplesToCollectThisStep / maxNumberOfCombinations
        #print("p = "+str(probabilityForDrawing))
        
        #adding training examples to the training set   
        for permutation in allSubsetsOfMaxLength(algorithmIndexes, len(algorithmIndexes)):
            if probabilityForDrawing < np.random.rand():
                #skip this example and continue witht he next one.
                continue
            inputIndexes = permutation
            #the counter set of all indexes that are not in inputIndexes:
            outputIndexes = [i for i in algorithmIndexes if not (i in inputIndexes)]

            tmp_score[outputIndexes] = 0
            tmp_costs[outputIndexes] = 0
            tmp_runs[outputIndexes] = 0
            tmp_score[inputIndexes] = scores[inputIndexes]
            tmp_costs[inputIndexes] = costs[inputIndexes]
            tmp_runs[inputIndexes] = runs[inputIndexes]
            np.concatenate((tmp_score, tmp_costs, tmp_runs), out=state, axis=0)
            
            a = ((state.T).copy(), (scores[outputIndexes]).copy(), (costs[outputIndexes]).copy(), outputIndexes.copy())
            self._trainingExamples.append(a)

            examplesCollected += 1
            #in the beginning, we collect more examples than maxExamplesCollected to fill up the storage quickly.
            if ( examplesCollected >= noOfExamplesToCollectThisStep ):
                break
        
        #using training examples from the collection to train the model
        currentNoOfExamples = len(self._trainingExamples)
        trainingSteps = int( currentNoOfExamples * min(1,( noOfDesiredTrainingSteps / maxExamplesToKeep )) )
        if (DEBUG_SHOW_PREDICTION_QUALITY):
            print("Examples Collected: "+str(examplesCollected)+" -> Examples used: "+str(trainingSteps))
        for i in range(trainingSteps):
            c = np.random.choice(len(self._trainingExamples))
            (features, scoreTargets, costTargets, targetIndexes) = self._trainingExamples[c]
            self._scoreModel.partial_fit(features, scoreTargets, targetIndexes)
            self._costModel.partial_fit(features, costTargets, targetIndexes)
            #remove the sample from the set
            self._trainingExamples = self._trainingExamples[:c] + self._trainingExamples[c+1:]

        '''
        for toBePredictedIndex in algorithmIndexes:
            remainingIndexes = algorithmIndexes[algorithmIndexes!=toBePredictedIndex]
        
            tmp_score[:] = 0
            tmp_costs[:] = 0
            tmp_runs[:] = 0
            
            for permutation in allSubsets(remainingIndexes):
                #remove target values from the features:
                tmp_score[permutation] = scores[permutation]
                tmp_costs[permutation] = costs[permutation]
                tmp_runs[permutation] = runs[permutation]
                np.concatenate((tmp_costs, tmp_runs), out=cost_state, axis=0)
                np.concatenate((tmp_score, tmp_runs), out=score_state, axis=0)
                self._costModel.partial_fit(cost_state.T, costs[toBePredictedIndex], toBePredictedIndex)
                self._scoreModel.partial_fit(score_state.T, scores[toBePredictedIndex], toBePredictedIndex)
                fitCounter += 1
        '''
        


    '''
        run the algorithm selection on one problem instance
    '''
    def process(self, budget, callback):
        #print("past performances: "+str(self._performance))
        #print("calls: "+str(self._runs))
        #set of the indexes of the remaining algorithms
        algorithms = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        costEstimators = self._costModel
        scoreEstimators = self._scoreModel
        remainingBudget = budget
        
        #STATE
        scores = np.zeros((np.shape(self._algorithms)[0],1))
        costs = np.zeros((np.shape(self._algorithms)[0],1))
        runs = np.zeros((np.shape(self._algorithms)[0],1), dtype=bool)
        state = np.concatenate((scores, costs, runs), axis=0)
        #cost_state = np.concatenate((costs, runs), axis=0)
        #score_state = np.concatenate((scores, runs), axis=0)
        
        explorationFactor = self._explorationFactor

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            #if ( np.any( self._runs < self._initialCollectionThreshold ) ):
            #estimatedCosts = np.ones(len(algorithms))
            #estimatedScores = np.ones(len(algorithms))

            estimatedCosts = costEstimators.predict(state.T)[0,algorithms]
            estimatedScores = scoreEstimators.predict(state.T)[0,algorithms]
            
            #np.nan_to_num(estimatedCosts, 0.0)
            #np.nan_to_num(estimatedScores, 0.0)
            
            #kick out choices that are probably too expensive
            estimatedScores[estimatedCosts > remainingBudget] = 0


            '''
                START AUC Optimized Choice
            '''
            estimatedAUCGain = estimatedScores * ( remainingBudget - estimatedCosts )
            weights = estimatedAUCGain
            weights[weights < 0] = 0
            bestIndex = np.argmax(weights)
            weights[:] = 0
            weights[bestIndex] = 1
            weights[:] += explorationFactor
            
            
            '''
            #minmax norm (also ensures there are no negative values)
            epsilon = 0.00001
            estimatedCosts = ( estimatedCosts - np.min(estimatedCosts) ) / (np.max(estimatedCosts) - np.min(estimatedCosts)) + epsilon
            estimatedScores = ( estimatedScores - np.min(estimatedScores) ) / (np.max(estimatedScores) - np.min(estimatedScores))
            
            
            
            #There are three considerations:
            #Choose the cheapest one (in the beginning) or the most efficient one to gather information
            #Later: choose the one with the highest estimated score that is still in the budget.
            costCriteria = np.max(estimatedCosts) - estimatedCosts
            efficiencyCriteria = estimatedScores / estimatedCosts
            efficiencyCriteria = ( efficiencyCriteria - np.min(efficiencyCriteria) ) / (np.max(efficiencyCriteria) - np.min(efficiencyCriteria))
            scoreCriteria = estimatedScores
            
            #at the beginning, we go for the cheapest evaluations, but as we progress, we go for the predicted best ones.
            #progress = 1.0 - ( remainingBudget / budget )
            #weights = (1.0 - progress) * efficiencyCriteria + progress * scoreCriteria
            weights = efficiencyCriteria
            
            #Again min/max normalization
            weights = weights - np.min(weights)
            weights = weights / np.max(weights)
            weights += self._explorationFactor
            #put more emphasis on the highest ones
            #weights = np.power(weights,2)
            '''
            
            #we are not considering thoose where we think we exceed the remaining Budget anyways.
            if ( np.all(weights==0) ):
                weights[:] = 1.0
            #Choose the algorithm randomly with a probability proportionally to it's historical performance
            weights = np.nan_to_num(weights, 0.0)
            if (np.sum(weights) == 0):
                weights[:] = 1
            
            
            #normalize such that they sum up to 1.0
            probabilities = weights / np.linalg.norm(weights, ord=1)
            
            #during the initialization phase, we choose the next algorithm based on the previously calculated weights. After that phase, we just choose the one with the highest weights.
            #if ( np.any( self._runs < self._initialCollectionThreshold ) ):
            selectedAlgorithm = np.random.choice(a = algorithms, replace = False, p = probabilities)
            #else:
            #    selectedAlgorithm = algorithms[np.argmax(probabilities)]
            
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
            algorithms = algorithms[algorithms != selectedAlgorithm]
                
            remainingBudget -= cost
            #print("algorithm:"+str(self._algorithms[ selectedAlgorithm ])+" score: "+str(score)+" cost: "+str(cost))
            
            

            if ( outOfBudget == False ):
                costs[selectedAlgorithm] = cost
                scores[selectedAlgorithm] = score
                runs[selectedAlgorithm] = True
                #np.concatenate((costs, runs), out=cost_state, axis=0)
                #np.concatenate((scores, runs), out=score_state, axis=0)
                np.concatenate((scores, costs, runs), out=state, axis=0)
            
        # DEBUGGING: Test prediction quality:
        if DEBUG_SHOW_PREDICTION_QUALITY:
            if ( self._debugDisplayCounter % 10 == 1 ):
                if ( np.any( self._runs >= self._initialCollectionThreshold ) ):
                    estimatedCosts = self._costModel.predict(state.T)
                    estimatedScores = self._scoreModel.predict(state.T)
                    mask = (runs[:,0]>0)
                    
                    ''' Zero Estimations '''
                    zeros = costs.copy()
                    zeros[:] = 0
                    np.concatenate((zeros, zeros, zeros), out=state, axis=0)
                    estimatedZeroCosts = self._costModel.predict(state.T)
                    estimatedZeroScores = self._scoreModel.predict(state.T)

                    print("Summary")
                    #print("cost state")
                    #print(str(cost_state.T))
                    print(" ### costs ### ")
                    print("Measured    "+str( costs[mask,0] ) )
                    print("Init. Est   "+str( estimatedZeroCosts[0,mask] )+ " mse: "+str(np.linalg.norm(costs[mask,0]-estimatedZeroCosts[0,mask])))
                    print("Final Est   "+str( estimatedCosts[0,mask] )+ " mse: "+str(np.linalg.norm(costs[mask,0]-estimatedCosts[0,mask])))
                    #print("score state")
                    #print(str(score_state.T))
                    print(" ### scores ### ")
                    print("Measured    "+str( scores[mask,0] ) )
                    print("Init. Est   "+str( estimatedZeroScores[0,mask] )+ " mse: "+str(np.linalg.norm(scores[mask,0]-estimatedZeroScores[0,mask])))
                    print("Final Est   "+str( estimatedScores[0,mask] )+ " mse:  "+str(np.linalg.norm(scores[mask,0]-estimatedScores[0,mask])))
                    print("runs")
                    print(self._runs)
                    print("exploration factor: "+str(self._explorationFactor))
            self._debugDisplayCounter += 1

        
        self.updateModels(scores.copy(), costs.copy(), runs.copy())
        
            
        #degrade the exploration factor.
        self._explorationFactor *= ( 1.0 - ( self._explorationDegrationFactor * (len(algorithms) / self._noOfAlgorithms ) ) )

