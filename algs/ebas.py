"""
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of the Online Ensemble based method for algorithm selection.
We use a basic linear online regression algorithm as baseline. We create one regressor per input variable and only let that regressor influence the final model if the specific input variable was there.
We also calculate a weight based on how accurate the single models performances where with respect to predicting the target values.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
"""

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
### custom (local) imports ### 
from .base.framework import AlgorithmSelector
from .base.framework import OutOfTimeException


    
class SparseEnsembleRegressionModel():
    """
    This ensemble model manages and combines multiple simple SGDRegressors into a flexible bigger multimodal regressor.
    
    Intended usage
    --------------
    Create an instance of this and tell how many input and how many output values you have. Call partial_fit() to update the internal regressors as fit
    
    Note: for each input <-> output combination, a new basic regressor is created
    
    """
    def __init__(self, xSize, ySize, baseModelCreator = None):
        self._xzSize = xSize
        self._yzSize = ySize
        
        if ( baseModelCreator is None ):
            def baseModelCreator():
                return SGDRegressor(penalty='l2', max_iter=1, learning_rate='optimal', eta0 = 0.1)
        
        #noOfInputAlgorithms+1 -> we also want a prediction if we got no input information at all.
        self._models = np.array([[baseModelCreator() for yi in range(ySize)] for xi in range(xSize+1)])
        self._weights = np.ones((xSize+1,ySize))
        

    def _weightAdjustment(self, oldWeight, predictedY, trueY):
        """
        Internal helper function to adjust the weights of the weighted average model based on the base model performances
        """
        newPointInfluence = 0.1
        return oldWeight * (1.0-newPointInfluence) + newPointInfluence * (1.0/(1.0 + np.abs(predictedY-trueY))) #note the (1.0 + np.abs) -> so the weights don't explode, it favors exploration of not so well known algorithms but thats ok.
    

    def partial_fit(self, X, xmask, y, ymask):
        """
        This method updates all the base models that could be updated based on which input and output values are available.
        To determine which values are actually really set and which are not, the xmask and ymask vectors are used (set to 1).
        
        Approach: we go through all the base estimators, adjust their ensemble weights and fit thoose whoose xmask and ymask are True.
        
        Parameters
        ----------
            X : numpy array, dtype = float
                feature vector np.shape(X) == (noOfInputAlgorithms,n)
            xmask : numpy array, dtype = boolean
                boolean mask for X that indicates which indexes contain actual values
            y : numpy array, dtype = float
                target value vector, np.shape(y) = (noOfOutputAlgorithms,n)
            ymask : numpy array, dtype = boolean
                boolean mask for y that indicates which labels contain actual values that should be fitted
        """
        assert len(xmask) == self._xzSize, "Error, mask xz unequal to _xzSize. Mask needs to be the same dimension as noOfInputAlgorithms."
        assert len(ymask) == self._yzSize, "Error, mask yz unequal to _xzSize. Mask needs to be the same dimension as noOfOutputAlgorithms."
        for yi in range(self._yzSize):
            if (ymask[yi] == True):
                for xi in range(self._xzSize):
                    if (xmask[xi] == True):
                        #Update the specific linear model. Note, before we fit the model, we check if it would have gotten this sample right to estimate how well it already generalizes.
                        (_X, _y) = self._format(X[xi], y[yi])
                        try:
                            self._weights[xi, yi] = self._weightAdjustment(self._weights[xi, yi], self._models[xi, yi].predict(_X), _y)
                            if (xi == yi):
                                self._weights[xi, yi] = 0 #dont use the own value for predictions.
                        except NotFittedError:
                            pass
                        
                        self._models[xi, yi].partial_fit(_X,_y)             
                (_X, _y) = self._format(X[0], y[yi])
                _X = np.zeros(np.shape(_X))
                
                #last, update the basic model that should only predict the average value as a baseline:
                try:
                    self._weights[self._xzSize, yi] = self._weightAdjustment(self._weights[self._xzSize, yi], self._models[self._xzSize, yi].predict(_X), _y)
                except NotFittedError:
                    pass
                self._models[self._xzSize, yi].partial_fit(_X,_y)
                
    def _format(self, X, y):
        #np.expand_dims(y,axis=1)
        return (np.expand_dims(X,axis=1), y)
        

    def predict(self, X, xmask, yindex):
        """
        This method creates a single prediction for one output dimension, based on a potentially sparse input set.
        
        Parameters
        ----------
        X : numpy array, dtype = float
            feature vector np.shape(X) == (noOfInputAlgorithms,n)
        xmask : numpy array, dtype = boolean
            boolean mask for X that indicates which indexes contain actual values
        yindex : integer
            index of the output dimension for which a prediction is returned.
        
        Returns
        -------
        float
            predicted value
            
        """
        ensembleSize = len(xmask)+1
        sumOfPredictions = 0
        sumOfWeights = 0
        for xzi in range(self._xzSize):
            if (xmask[xzi] == True):
                try:
                    pass
                    sumOfPredictions += self._models[xzi, yindex].predict(np.expand_dims(X[xzi],axis=1)) * self._weights[xzi, yindex] 
                    sumOfWeights += self._weights[xzi, yindex] 
                except NotFittedError:
                    pass
        _X = np.expand_dims(np.zeros(np.shape(X[0])),axis=1)
        try:
            sumOfPredictions += self._models[self._xzSize, yindex].predict(_X) * self._weights[self._xzSize, yindex] 
            sumOfWeights += self._weights[self._xzSize, yindex] 
        except NotFittedError:
            pass
        
        if (sumOfWeights==0):
            return 1
        return sumOfPredictions / sumOfWeights
        
    def predictMultiple(self, X, xmask, yindexes):
        """
        This method creates predictions for all via yindex requested output dimensions, based on a potentially sparse input set.
        
        Parameters
        ----------
            X : numpy array, dtype = float
                feature vector np.shape(X) == (noOfInputAlgorithms,n)
            xmask : numpy array, dtype = boolean
                boolean mask for X that indicates which indexes contain actual values
            yindexes : numpy array, dtype = boolean
                integer index indicating the output values for which a prediction is required.
                
        Returns
        -------
        numpy array, dtype = float, np.shape = np.shape(yindexes)
        """

        result = np.zeros(len(yindexes))
        counter = 0
        for yi in yindexes:
            result[counter] = self.predict(X, xmask, yi)
            counter += 1
        return result
    
    def getFeatureImportance(self):
        """
        Returns the median weight for a specific input feature as an estimate of the importance of that feature for the prediction quality of others.
        
        Returns
        -------
        numpy array, dtype = float
            a 1d-array containing the average weights of the internal ensemble model for each input feature (therefore the same dimensions)
        """

        #return without the base weights, because they are not aligned to any input feature.
        return (np.median( self._weights[:-1], axis = 1))
        
        

class EnsembleBasedAlgorithmSelector(AlgorithmSelector):
            
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
        self._explorationFactor = 0.1 #how much weight we put in exploration
        self._explorationDegrationFactor = 0.0 #hyperparameter between 0 and 1
        self._initialCollectionThreshold = 1
        
        #initialize a score model for each algorithm
        self._scoreModel = SparseEnsembleRegressionModel(xSize = self._noOfAlgorithms, ySize = self._noOfAlgorithms)
        self._costModel = SparseEnsembleRegressionModel(xSize = self._noOfAlgorithms, ySize = self._noOfAlgorithms)
        

        self._lowestCost = None
        self._debugDisplayCounter = 0
        

    def updateModels(self, scores, costs, runs):
        self._scoreModel.partial_fit(X=scores, xmask=runs, y=scores, ymask=runs)
        self._costModel.partial_fit(X=costs, xmask=runs, y=costs, ymask=runs)


    '''
        run the algorithm selection on one problem instance
    '''
    def process(self, budget, callback):
        #print("past performances: "+str(self._performance))
        #print("calls: "+str(self._runs))
        #set of the indexes of the remaining algorithms
        algorithms = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        remainingBudget = budget
        
        #STATE
        scores = np.zeros((np.shape(self._algorithms)[0],1))
        costs = np.zeros((np.shape(self._algorithms)[0],1))
        runs = np.zeros((np.shape(self._algorithms)[0],1), dtype=bool)

        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            if ( np.any( self._runs < self._initialCollectionThreshold ) ):
                estimatedCosts = np.ones(len(algorithms))
                estimatedScores = np.ones(len(algorithms))
            else:
                estimatedCosts = self._costModel.predictMultiple(X = costs, xmask = runs, yindexes = algorithms)
                estimatedScores = self._scoreModel.predictMultiple(X = scores, xmask = runs, yindexes = algorithms)
                #estimatedScores = self._scoreModel.predictMultiple(X = scores.copy(), xmask = np.zeros(np.shape(runs)), yindexes = algorithms)
            
            #make sure there are no negative values:
            estimatedCosts = np.abs(estimatedCosts)
            estimatedCosts[estimatedCosts <= 0] = 10000.0 #epsilon
            estimatedScores[estimatedScores < 0] = 0


            #Old score/costs strategy:
            estimatedPerformances = estimatedScores / estimatedCosts
            
            
            #boost evaluations that are important for the performance prediction:
            featureImportance = self._costModel.getFeatureImportance() + self._scoreModel.getFeatureImportance()
            featureImportance = featureImportance[algorithms]#filter for the remaining algorithms
            featureImportance = featureImportance / np.linalg.norm(featureImportance)
            estimatedPerformances = np.multiply(estimatedPerformances, featureImportance)
                
            estimatedPerformances[estimatedCosts > budget] = 0
            
            #New strategy
            #estimatedScores[estimatedCosts > remainingBudget] = 0 #sort out the ones that are likely not to finish within the budget
            #estimatedPerformances = estimatedScores - np.min(estimatedScores) #adjust range (worst ones gets 0 chance to run)

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
            
            #add to history so we can learn from it:
            if ( outOfBudget == False ):
                costs[selectedAlgorithm] = cost
                scores[selectedAlgorithm] = score
                runs[selectedAlgorithm] = True
            
        
            
        # DEBUGGING: Test prediction quality:
        '''
        if ( self._debugDisplayCounter % 100 == 0 ):
            if ( np.any( self._runs >= self._initialCollectionThreshold ) ):
            
                allAlgorithmIndexes = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
                estimatedCosts = self._costModel.predictMultiple(X = costs.copy(), xmask = runs, yindexes = allAlgorithmIndexes)
                estimatedScores = self._scoreModel.predictMultiple(X = scores.copy(), xmask = runs, yindexes = allAlgorithmIndexes)
                
                avgEstScores = self._scoreModel.predictMultiple(X = scores.copy(), xmask = np.zeros(np.shape(runs)), yindexes = allAlgorithmIndexes)

                mask = (runs[:,0]>0)
                print("Summary")
                print("Measured vs estimated costs")
                print(costs[mask,0])
                print(estimatedCosts[mask])
                print("Measured vs estimated scores vs. base est")
                print(scores[mask,0])
                print(estimatedScores[mask])
                print(avgEstScores[mask])
                print("runs")
                print(self._runs[mask])
                print("exploration factor: "+str(self._explorationFactor))
                
                #print(self._costModel._weights)
                #print(self._scoreModel._weights)
        self._debugDisplayCounter += 1
        '''
        self.updateModels(scores.copy(), costs.copy(), runs.copy())
        

        #degrade the exploration factor.
        self._explorationFactor *= ( 1.0 - ( self._explorationDegrationFactor / self._noOfAlgorithms ) )

