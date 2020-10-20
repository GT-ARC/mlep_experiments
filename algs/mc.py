'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the implementation of a monte carlo baseline version.

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


class MonteCarloAlgorithmSelector(AlgorithmSelector):
    def __init__(self, algorithms):
        self._algorithms = algorithms
        self._noOfAlgorithms = np.shape(self._algorithms)[0]
        
    def process(self, budget, callback):
        algorithms = np.arange(start=0,stop=self._noOfAlgorithms,step=1, dtype=np.int32)
        remainingBudget = budget
        while ( ( remainingBudget > 0 ) & ( len(algorithms)>0 ) ):
            selectedAlgorithm = np.random.choice(a = algorithms, replace = False)
            #remove the selected algorithm from the local set:
            algorithms = algorithms[algorithms != selectedAlgorithm]
            #run the algorithm and get the budget and cost back
            try:
                ( cost, score ) = callback( self._algorithms[ selectedAlgorithm ])
            except OutOfTimeException:
                #end the while loop
                remainingBudget = 0
                break

