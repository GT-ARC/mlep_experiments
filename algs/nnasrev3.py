'''

Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the Neural Network Algorithm Selector method, based on reinforcement learning.

The system contains the following parts:

A state function s(s_raw) that aggregates the current state of the algorithm selection process for a single problem instance into a meaningfull state vector.
    Contains:
        Remaining budget
        Scores and Costs of previous evaluations on the current problem instance
        

The current algorithm selection policy P(s) -> a, where s is the state vector and a is a vector of continous values indicating the priority of which algorithm should be evaluated next.


A value function v(s_raw) that returns the absolute true value achieved so far (=historical values plus future values based on remaining budget * best score found so far)
Note: this is a pessimistic lower bound under the assumption that no better algorithm will be found within the remaining budget.


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


class NNModelWrapper():
    baseNetwork = None
    
    def __init__(self, inputSize, outputSize):
        net = nn.Sequential()
        net.add(
            self.createBaseNetwork(inputSize, outputSize),
            nn.Dense(outputSize, activation = "sigmoid", flatten=True),
        )
        net.initialize(init=init.Xavier(), force_reinit = False) #force_reinit 
        self.model = net
        self.objectiveFunction = gluon.loss.L2Loss(batch_axis = 1)
        self.trainer = gluon.Trainer(net.collect_params(), 'adagrad')
        #self.trainer = gluon.Trainer(net.collect_params(), 'adam')
        #self.trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.00001, 'momentum':0.0})
        self.avgLoss = 0
        self.avgLoss2 = 0
        self.lossCounter = 0
        
    def createBaseNetwork(self, inputSize, outputSize):
        if (NNModelWrapper.baseNetwork is None):
            net = nn.Sequential()
            net.add(
                #nn.Dense(inputSize, activation = "relu"),
                #nn.Dense(10*inputSize, activation = "relu"),
                nn.Dense(3, activation = "relu"),
                #nn.Dense(outputSize, activation = "relu"),
                #nn.Dense(1, flatten=True),
                #nn.Dense(1*noOfAlgorithms, activation = "relu"),
                #nn.GlobalAvgPool1D(),
                #nn.Flatten()
            )
            NNModelWrapper.baseNetwork = net
        return NNModelWrapper.baseNetwork
    
    
    def partial_fit(self, X, ytrue, positions):
        '''
        X = a single feature vector
        y = a vector of target values
        positions = the indexes of the target vector y that are supposed to be used to update the models (all others will not influence the weight update)
        '''
        if len(positions) == 0:
            return
        '''
        We only use some y labels and set the loss for all the others to zero.
        '''
        X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        
        y = ((self.model(X)[0,:]).abs()) #* 0.9 #we set y to the current predictions of the model.
        #y = y * (1.0 - (0.1*np.random.rand())) #move y towards zero a tiny bit.
        #print( np.shape(y[positions]))
        #print( np.shape(ytrue[positions]))
        #print("true values: "+str(ytrue[positions][0]))
        y[positions] = ytrue[positions][0] 
        
        with autograd.record():
            output = self.model(X)
            loss = self.objectiveFunction(output, y)

        loss.backward()
        self.trainer.step(1)
        #self.trainer.step(np.shape(y)[0])
        
        loss2 = self.objectiveFunction(self.model(X), y)
        self.avgLoss += loss.asnumpy()
        self.avgLoss2 += loss2.asnumpy()
        self.lossCounter += 1
        return loss
        
    def partial_fit2(self, X, ytrue, positions):
        '''
        X = a single feature vector
        y = a vector of target values
        positions = the indexes of the target vector y that are supposed to be used to update the models (all others will not influence the weight update)
        '''
        batch = True
        
        if len(positions) == 0:
            return
        '''
        We only use some y labels and set the loss for all the others to zero.
        '''
        (no_of_samples, feature_dim, _ ) = np.shape(X)
        X = np.reshape(X,(no_of_samples, feature_dim))
        
        (no_of_samples, output_dim, _ ) = np.shape(ytrue)
        ytrue = np.reshape(ytrue,(no_of_samples, output_dim))
        
        #X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        
        #print("X: "+str( np.shape(X) ) )
        y = self.model(X)
        #y = y / nd.max(y.abs())#/nd.max(y.abs())) #* 0.9 #we set y to the current predictions of the model.
        #print("y: "+str( np.shape(y) ))
        #print("ytrue: "+str( np.shape(ytrue) ))
        #print("positions: "+str( np.shape(positions) ) )

        for si in range(no_of_samples):
            y[si,positions[si]] = ytrue[si,positions[si]]
        
        steps = no_of_samples
        
        with autograd.record():
            output = self.model(X)
            loss = self.objectiveFunction(output, y)
            '''
            si = 0#np.random.randint(no_of_samples)
            for si in range(no_of_samples):
                #print("si: "+str(si)+" -> "+str(positions[si]))
                #print("X: "+str(X[si].asnumpy()))
                #print("example output: "+str(output[si,positions[si]]))
                #print("example output: "+str(output[si,:]))
                #print("example true y: "+str(ytrue[si,positions[si]]))
                #print("example true y: "+str(ytrue[si,:]))
            '''
            if batch:
                loss = nd.mean(loss, axis = 0)
                steps = 1
        loss.backward()
 
        self.trainer.step(steps)
        
        loss2 = self.objectiveFunction(self.model(X), y)
        if batch:
            loss2 = nd.mean(loss2, axis = 0)
            
        self.avgLoss += np.sum(loss.asnumpy())
        self.avgLoss2 += np.sum(loss2.asnumpy())
        self.lossCounter += steps
        return self.avgLoss / self.lossCounter
        
    def printAvgLossAndReset(self):
        if ( self.lossCounter >= 100):
            print("avg loss: "+str(np.sum(self.avgLoss)/self.lossCounter))
            print("avg loss2: "+str(np.sum(self.avgLoss2)/self.lossCounter))
            self.avgLoss = 0
            self.avgLoss2 = 0
            self.lossCounter = 0
        return 
        
    def predict(self, X):
        X = np.expand_dims(X, axis = 0)
        X = nd.array(X)
        y = self.model(X).asnumpy()
        return y
        
    def score(self, X, y):
        return self.objectiveFunction(self.predict(X), y)
            
'''
    Basic approach:
    
    When starting a new process, we update our model with the past data.

'''

SINGLE_LOOKAHEAD = 1 #singe example fit.
MULTI_LOOKAHEAD = 2 #batch leaning of SINGLE_LOOKAHEAD
MULTI2_LOOKAHEAD = 3 #we provide multiple AUC values to train for instead of just one.
STRICT_ONPOLICY = 4

DECAYING_CHANCE_FOR_RANDOM_PICKS = 1
RANDOM_PICKS_WEIGHTED_BY_MODEL_CERTAINTY = 2

        
class NeuralNetworkAlgorithmSelectorR3(AlgorithmSelector):
            
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

        
        #initialize a score model for each algorithm
        #model = PassiveAggressiveRegressor
        #model = SGDRegressor
        self._policyModel = NNModelWrapper(self._noOfAlgorithms*2+1,self._noOfAlgorithms) #NNModelWrapper(self._noOfAlgorithms*3+3,self._noOfAlgorithms) 
        self._lowestCost = None
        self._noOfPolicyCalls = 0
        
        self.updatepolicy = STRICT_ONPOLICY
        self.randomselectionstrategy = DECAYING_CHANCE_FOR_RANDOM_PICKS
        
        self._trainingExamples = list()

    def calculateAUC(self, scores, costs, runningOrder, initial_budget):
        '''
        this is the pessimistic value function v(s_raw) as described at the top of this file.
        '''
        remainingBudget = initial_budget
        lastScore = 0
        totalScore = 0
        for ai in runningOrder:
            cost = costs[ai]
            if (cost > remainingBudget):
                cost = remainingBudget
            remainingBudget -= cost
            totalScore += lastScore * cost
            lastScore = max( lastScore, scores[ai]) #keep the best score so far
            
            if (remainingBudget <= 0):
                return totalScore
                
        totalScore += lastScore * remainingBudget#use up the remaining budget
        return totalScore
        
    def stateFunction(self, scores, costs, runs, remainingBudget, oldstate = None):
        '''
        this is the state function s(raw_state) as described at the top of this file.
        '''

        #v = [np.reshape([remainingBudget, max_score, max_cost], (3,1))]              
        #v = (scores, costs, runs, np.reshape([remainingBudget, max_score, max_cost], (3,1))) #adjust also the model input size in the __init__ function fi you modify this.
        v = (scores, costs, np.reshape([remainingBudget,1], (2,1))) #the static one is for the network to be able to add a bias
        #v = (scores, costs)
        if ( oldstate is None ):
            return np.concatenate(v, axis=0)
        np.concatenate(v, out=oldstate, axis=0)
        return oldstate
        
        
    def policyFunction(self, state):
        '''
        policy function that decides which evaluation/algorithm should be run next
        '''
        weights = self._policyModel.predict(state.T)
        #remove nan's.
        weights = np.nan_to_num(weights, 0.0)
        #if all weights are zero, set them to a uniform value
        if ( np.all(weights==0) ):
            weights[:] = 1.0
        if (np.sum(weights) == 0):
            weights[:] = 1
        
        self._noOfPolicyCalls += 1
        
        
        if ( self.randomselectionstrategy == DECAYING_CHANCE_FOR_RANDOM_PICKS ):
            p = 0.05 + self._noOfAlgorithms / ( self._noOfAlgorithms + self._noOfPolicyCalls )
            if np.random.rand()>p:
                weights = np.random.rand(*np.shape(weights))
                
        elif ( self.randomselectionstrategy == RANDOM_PICKS_WEIGHTED_BY_MODEL_CERTAINTY ):
            w = weights[0,:]
            w = w / np.sum(w)
            selectedAlgorithm = np.random.choice(a = len(w), replace = False, p = w)
            weights[0,:] = 0
            weights[0,selectedAlgorithm] = 1.0
        '''
        #this would be the optimal strategy for synthetic benchmark d, for debugging.
        weights[0,:] = 0
        if state[6] == 0.9:
            weights[0,0] == 1
        elif state[0] > 0:
           weights[0,1] == 1
        else:
            weights[0,2] = 1
        '''
        return weights
    
    def updatePolicy(self, scores, costs, runningOrder, initial_budget):
        '''
        This function updates the policy model with new training examples.
        '''   
        #the scores, costs, runningOrder and initial_budget are created by evaluating the current policy e.g. the current model.
        if (len(runningOrder)==0):
            return
            
        runningOrder = np.array(runningOrder)
        
        tmp_runs = np.zeros((self._noOfAlgorithms,1), dtype=bool)
        state = self.stateFunction(scores, costs, tmp_runs, initial_budget)
        
        #we have two options: we can train to predict the final score achieved
        #or we predict the 1-step ahead score.
        
        #print("scores: "+str(scores[runningOrder]))
        #print("costs: "+str(costs[runningOrder]))
        #print("runningOrder: "+str(runningOrder))
        


        if (self.updatepolicy == SINGLE_LOOKAHEAD):
            #one step ahead method: we pretend going through the very same selection process and provide examples of states and the score achieved after following the policy one step further.
            #if noOfGames == 1 then this is an on-policy method. otherwise, only the first round is on-policy, the other ones are random variations based on the same objects in the sequence.
            noOfGames = 1#len(runningOrder)
            for i in range(noOfGames):
                finalAUC = self.calculateAUC(scores, costs, runningOrder, initial_budget)
                for noOfAlgorithms in range(len(runningOrder)-1):
                    tmp_score = scores.copy()
                    tmp_costs = costs.copy()
                    localRunningOrder = runningOrder[:noOfAlgorithms+1] #running order including the NEXT algorithm choice.
                    #print("localRunningOrder: "+str(localRunningOrder))
                    #print("localRunningOrder[-1]: "+str(localRunningOrder[-1]))
                    #calculate the future AUC after choosing the next algorithm:
                    auc_scores = np.zeros(np.shape(scores))
                    #auc_scores[localRunningOrder[-1]]  = finalAUC
                    #auc_scores[localRunningOrder[-1]]  = self.calculateAUC(tmp_score, tmp_costs, localRunningOrder, initial_budget)
                    
                    #we estimate the max achievable AUC for this node by randomly permuting the remaining orders, then calculating the AUC of each and use the max.
                    remaining = runningOrder[noOfAlgorithms+1:]
                    remaining = remaining.copy()
                    estimatedAUC = self.calculateAUC(tmp_score, tmp_costs, localRunningOrder, initial_budget)
                    for i in range(len(remaining)):
                        np.random.shuffle(remaining)
                        if len(localRunningOrder) == 0:
                            tmp_order = remaining
                        elif len(remaining) == 0:
                            tmp_order = localRunningOrder
                        else:
                            tmp_order = np.concatenate((localRunningOrder,remaining),axis=0)
                        tmp_auc = self.calculateAUC(scores, costs, tmp_order, initial_budget)
                        if (tmp_auc > estimatedAUC):
                            estimatedAUC = tmp_auc
                    
                    auc_scores[localRunningOrder[-1]] = estimatedAUC
                    
                    #print("auc_scores[localRunningOrder[-1]] : "+str(auc_scores[localRunningOrder[-1]] ))
                    #now we only deal with the past and current situation. So we update the localRunningOrder by removing the "future" algorithm pick.
                    localRunningOrder = runningOrder[:noOfAlgorithms]
                    setToZeroIndexes = [a for a in range(len(tmp_score)) if (a not in localRunningOrder)]
                    tmp_score[setToZeroIndexes] = 0
                    tmp_costs[setToZeroIndexes] = 0
                    remainingBudget = initial_budget - np.sum( tmp_costs )
                    tmp_runs[:] = False
                    tmp_runs[localRunningOrder] = True
                    state = self.stateFunction(tmp_score, tmp_costs, tmp_runs, remainingBudget, state)
                    #print("auc_scores: "+str(auc_scores))
                    #print("noOfAlgorithms: "+str(noOfAlgorithms))
                    #print("[runningOrder[noOfAlgorithms]]: "+str([runningOrder[noOfAlgorithms]]))
                    self._policyModel.partial_fit(state, auc_scores, [runningOrder[noOfAlgorithms]])
                #after we first used the actual selected runningOrder, we now shuffle the running order to create alternative scenarios.
                np.random.shuffle(runningOrder)

        elif (self.updatepolicy == MULTI_LOOKAHEAD):
            #one step ahead method: we pretend going through the very same selection process and provide examples of states and the score achieved after following the policy one step further.
            #if noOfGames == 1 then this is an on-policy method. otherwise, only the first round is on-policy, the other ones are random variations based on the same objects in the sequence.
            noOfGames = 1#len(runningOrder)
            state_list = []
            auc_scores_list = []
            updateIndexes_list = []

            for i in range(noOfGames):
                finalAUC = self.calculateAUC(scores, costs, runningOrder, initial_budget)
                for noOfAlgorithms in range(len(runningOrder)-1):
                    tmp_score = scores.copy()
                    tmp_costs = costs.copy()
                    localRunningOrder = runningOrder[:noOfAlgorithms+1] #running order including the NEXT algorithm choice.
                    auc_scores = np.zeros(np.shape(scores))

                    #we estimate the max achievable AUC for this node by randomly permuting the remaining orders, then calculating the AUC of each and use the max.
                    remaining = runningOrder[noOfAlgorithms+1:]
                    remaining = remaining.copy()
                    estimatedAUC = self.calculateAUC(tmp_score, tmp_costs, localRunningOrder, initial_budget)
                    for i in range(len(remaining)):
                        np.random.shuffle(remaining)
                        if len(localRunningOrder) == 0:
                            tmp_order = remaining
                        elif len(remaining) == 0:
                            tmp_order = localRunningOrder
                        else:
                            tmp_order = np.concatenate((localRunningOrder,remaining),axis=0)
                        tmp_auc = self.calculateAUC(scores, costs, tmp_order, initial_budget)
                        if (tmp_auc > estimatedAUC):
                            estimatedAUC = tmp_auc
                            
                    auc_scores[localRunningOrder[-1]] = estimatedAUC
                    
                    #print("auc_scores[localRunningOrder[-1]] : "+str(auc_scores[localRunningOrder[-1]] ))
                    #now we only deal with the past and current situation. So we update the localRunningOrder by removing the "future" algorithm pick.
                    localRunningOrder = runningOrder[:noOfAlgorithms]
                    setToZeroIndexes = [a for a in range(len(tmp_score)) if (a not in localRunningOrder)]
                    tmp_score[setToZeroIndexes] = 0
                    tmp_costs[setToZeroIndexes] = 0
                    remainingBudget = initial_budget - np.sum( tmp_costs )
                    tmp_runs[:] = False
                    tmp_runs[localRunningOrder] = True 
                    state = self.stateFunction(tmp_score, tmp_costs, tmp_runs, remainingBudget, state)
                    #print("auc_scores: "+str(auc_scores))
                    #print("noOfAlgorithms: "+str(noOfAlgorithms))
                    #print("[runningOrder[noOfAlgorithms]]: "+str([runningOrder[noOfAlgorithms]]))
                    state_list.append(state.copy())
                    auc_scores_list.append(auc_scores)
                    updateIndexes_list.append([runningOrder[noOfAlgorithms]])
                #after we first used the actual selected runningOrder, we now shuffle the running order to create alternative scenarios.
                np.random.shuffle(runningOrder)
            self._policyModel.partial_fit2(state_list, auc_scores_list, updateIndexes_list)
        elif ( self.updatepolicy == MULTI2_LOOKAHEAD ):
            #in this method, we try to predict the final AUC values.
            noOfGames = 1#len(runningOrder)
            state_list = []
            auc_scores_list = []
            updateIndexes_list = []
            

            for i in range(noOfGames):
                #finalAUC = self.calculateAUC(scores, costs, runningOrder, initial_budget)
                for ai in range(len(runningOrder)-1):
                    #ai indicates how far we follow the original running order.
                    #based on ai, we calculate the current state representation.
                    tmp_score = scores.copy()
                    tmp_costs = costs.copy()
                    localRunningOrderWithoutNext = runningOrder[:ai].copy()
                    remainingRunningOrderWithNext = runningOrder[ai:].copy()
                    auc_scores = np.zeros(np.shape(scores))
 
                    #now, we calculate an approximation of the final AUC for each of the remaining choices:
                    for i in range(len(remainingRunningOrderWithNext)):
                        ai2 = remainingRunningOrderWithNext[i]
                        rest = np.concatenate((remainingRunningOrderWithNext[:i],remainingRunningOrderWithNext[i:]), axis = 0)
                        bestAUC = 0
                        for i in range(len(remainingRunningOrderWithNext)):
                            np.random.shuffle(rest)
                            auc = self.calculateAUC(tmp_score, tmp_costs, np.concatenate((localRunningOrderWithoutNext,[ai2],rest),axis=0) , initial_budget)
                            if (auc > bestAUC ):
                                bestAUC = auc
                        auc_scores[ai2] = bestAUC
                        
                    tmp_score[remainingRunningOrderWithNext] = 0
                    tmp_costs[remainingRunningOrderWithNext] = 0
                    
                    remainingBudget = initial_budget - np.sum( tmp_costs )
                    tmp_runs[:] = False
                    tmp_runs[localRunningOrderWithoutNext] = True
                    
                    if np.any(auc_scores!=0):
                        auc_scores = (auc_scores - np.min(auc_scores)) / (np.max(auc_scores) - np.min(auc_scores))
                        #auc_scores = auc_scores / np.max(auc_scores)
                    
                    #auc_scores = 2 * auc_scores
                    #auc_scores = auc_scores - 1.0
                    

                    
                    #this sample is only interesting if it teaches the model a difference of AUC. In most of the end cases, when the choice does not affect the AUC any longer, the choice is meaningless.
                    if (np.sum(auc_scores) < len(remainingRunningOrderWithNext)):
                        state = self.stateFunction(tmp_score, tmp_costs, tmp_runs, remainingBudget, state)
                        state_list.append(state.copy()) 
                        auc_scores_list.append(auc_scores)
                        updateIndexes_list.append(remainingRunningOrderWithNext)
                #after we first used the actual selected runningOrder, we now shuffle the running order to create alternative scenarios.
                np.random.shuffle(runningOrder)
            self._policyModel.partial_fit2(state_list, auc_scores_list, updateIndexes_list)
        elif ( self.updatepolicy == STRICT_ONPOLICY ):
            state_list = []
            auc_scores_list = []
            updateIndexes_list = []
            finalAUC = self.calculateAUC(scores, costs, runningOrder, initial_budget)
            
            finalScore = np.max(scores)

            for ai in range(len(runningOrder)-1):
                nextAlgorithm = runningOrder[ai]
                tmp_score = scores.copy()
                tmp_costs = costs.copy()
                auc_scores = np.zeros(np.shape(scores))
                
                localRunningOrderWithoutNext = runningOrder[:ai].copy()
                remainingRunningOrderWithNext = runningOrder[ai:].copy()
                remainingRunningOrderWithoutNext = runningOrder[ai+1:].copy()
                tmp_score[remainingRunningOrderWithNext] = 0
                tmp_costs[remainingRunningOrderWithNext] = 0
                remainingBudget = initial_budget - np.sum( tmp_costs ) 
                state = self.stateFunction(tmp_score, tmp_costs, tmp_runs, remainingBudget, state)
                
                #auc_scores[nextAlgorithm] = finalScore
                auc_scores[nextAlgorithm] = finalAUC
                
                #print("ai: "+str(ai))
                #print("State: "+str(state))
                #print("scores: "+str(auc_scores))
                #print("remaining running order with next: "+str(remainingRunningOrderWithNext))
                
                state_list.append(state.copy()) 
                auc_scores_list.append(auc_scores)
                updateIndexes_list.append([nextAlgorithm])
            self._policyModel.partial_fit2(state_list, auc_scores_list, updateIndexes_list)

            
        
        self._policyModel.printAvgLossAndReset()
        



    def process(self, budget, callback):
        '''
            run the algorithm selection on one problem instance
        '''
        #print("past performances: "+str(self._performance))
        #print("calls: "+str(self._runs))
        #set of the indexes of the remaining algorithms
        algorithms = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        policyEstimators = self._policyModel
        remainingBudget = budget
        
        #STATE
        scores = np.zeros((np.shape(self._algorithms)[0],1))
        costs = np.zeros((np.shape(self._algorithms)[0],1))
        runningOrder = list()
        runs = np.zeros((np.shape(self._algorithms)[0],1), dtype=bool)
        state = self.stateFunction(scores, costs, runs, remainingBudget)

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            algorithmPriorities = self.policyFunction(state)
            #keep only the ones that we have not evaluated so far:
            algorithmPriorities = algorithmPriorities[0,algorithms]
            #select the algorithm that has the highest policy value
            selectedAlgorithm = algorithms[ np.argmax(algorithmPriorities) ]

            #run the algorithm and get the budget and cost back
            self._runs[selectedAlgorithm] += 1
            outOfBudget = False
            try:
                ( cost, score ) = callback( self._algorithms[ selectedAlgorithm ])
            except OutOfTimeException:
                cost = remainingBudget
                score = 0
                outOfBudget = True
            
            #if we exceeded the budget, this run does not count.
            if (remainingBudget < cost ):
                cost = remainingBudget
                score = 0
                outOfBudget = True


            #remove the selected algorithm from the local set, as well as from the estimator sets
            algorithms = algorithms[algorithms != selectedAlgorithm]
                
            remainingBudget -= cost
            #print("algorithm:"+str(self._algorithms[ selectedAlgorithm ])+" score: "+str(score)+" cost: "+str(cost))
            

            #if ( outOfBudget == False ):
            costs[selectedAlgorithm] = cost
            scores[selectedAlgorithm] = score
            runs[selectedAlgorithm] = True
            runningOrder.append(selectedAlgorithm)
            state = self.stateFunction(scores, costs, runs, remainingBudget, state)
            

        self.updatePolicy(scores.copy(), costs.copy(), runningOrder, initial_budget = budget)

