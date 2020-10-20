'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains a function test for algorithm selection methods.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###
import math, os
### 3rd party imports (from packages, the environment) ### 
import numpy as np
### custom (local) imports ### 
from algs.base.evaluation import AlgorithmSelectionProcess, oracle, calculateAUCViaCallback, AlgorithmSelectionProcessRecorder


'''
This synthetic benchmark evaluates AlgorithmSelectors on a bunch of generical algorithms and problems.

The algorithms are designed with the following properties:
The score is normed to be between (1,10].
The runtime is normed to be between [1,10].
There is no guarantee that any algorithm on a given problem achieves the maximum or minimum score.
The performance (score and runtime) of the algorithms is related (can be exploited)
There is not per se an average best algorithm.

= Input =
selectors - list of tuples, each consisting of a name and a reference to an AlgorithmSelector class.
    Note, the benchmark will later on create instances of the algorithm selector classes and set the number of algorithms.
alg_n - number of synthetic algorithms created and used in the benchmark.
problem_n - number of problems to evaluate on. Should be much larger than the number of algorithms.
budgetfactor - how much time with respect to the overall required time to run all algorithms is provided (per problem). Setting this to 1 gives enough budget to run all the algorithms, so only values below 1.0 make sense to create a challenge.

= Output =
While running: using an oracle to calculate the highest practically achievable score (assuming to choose the best algorithm that can finish within the given time budget).
The average Area under the curve with respect to the highest observed AUC of the other competing methods.

= Example usage =
selectors = list()
selectors.append( ("MonteCarlo", MonteCarloAlgorithmSelector) )
selectors.append( ("Conservative", ConservativeAlgorithmSelector) )   
benchmark_synthetic_a(selectors = selectors, alg_n = 100, problem_n = 100000 )
'''
def benchmark_synthetic_a(selectors, alg_n = 100, problem_n = 10000, budgetfactor = 0.25):
    algorithms = np.arange(0,alg_n) #x algorithms
    np.random.shuffle(algorithms) #shuffle the algorithms in place to destroy their smoothness
    problems = np.random.rand(problem_n)*len(algorithms) #random vector of size 10000 with numbers between 0 and 10.
    
    noOfSelectors = len(selectors) 
    
    #initialize the selector instances from the provided classes
    selectors2 = list()
    selectorNames = list()
    for s in selectors:
        selectors2.append((s[0],s[1](algorithms = algorithms)))
        selectorNames.append(s[0])
    selectors = selectors2
    
    
    summedScores_auc = np.zeros(noOfSelectors)
    summedScores_finalscore = np.zeros(noOfSelectors)
    decayingScores_auc = np.zeros(noOfSelectors)
    decayingScores_finalscore = np.zeros(noOfSelectors)
    decayFactor = 0.1

    problemCounter = 0
    printDecayStatsAfterNProblems = 10
    printSummedStatsEveryNProblems = 10
    
    resultRecorder = AlgorithmSelectionProcessRecorder(experiment_id = os.path.basename(__file__), noOfAlgorithms = noOfSelectors, noOfProblems = problem_n, namesOfAlgorithms = selectorNames )

    for p in problems:
        problemCounter += 1
        def callback(algorithm):
            simulatedCost = (algorithm % 10)+1 #range: (1, 10), linear distributed
            simulatedScore = (1.0 / ( 1 + np.abs(p - algorithm) ))*9 + 1# 1-10, based on their distance
            return ( simulatedCost, simulatedScore )
        
        budget = len(algorithms)*10*budgetfactor#max cost is 10, therefore 10*0.5 is about 50% of the budget required to go through half of the algorithms.
        
        #print("=== Problem: "+str(p)+" b:"+str(budget)+" ===")
        
        (oCost, oScore, oAlgorithm, oSequence, oAUC) = oracle(algorithms, callback, budget)
        
        #print("oracle oAUC: "+str(oAUC)+" sequence: "+str(oSequence))

        counter = 0
        for (selectorName, selector) in selectors:
            asprocess = AlgorithmSelectionProcess( budget = budget, callback = callback, measuretime = False )
            selector.process(budget = budget, callback = asprocess.callback)
            (bScore, bAlgorithm) = asprocess.getBest()
            nAUC = asprocess.normalizedAUC( maxAUCScore = oAUC )
            nBestScore = bScore / oScore
            
            resultRecorder.addRecord(counter, problemCounter-1, {'nauc':nAUC, 'nBestScore':nBestScore})
            summedScores_auc[counter] += nAUC
            summedScores_finalscore[counter] += nBestScore
            decayingScores_auc[counter] = decayingScores_auc[counter] * (1.0-decayFactor) + decayFactor * nAUC
            decayingScores_finalscore[counter] = decayingScores_finalscore[counter] * (1.0-decayFactor) + decayFactor * nBestScore
            counter += 1

            
        printDecayingPerformances = (problemCounter % printDecayStatsAfterNProblems == 0 )
        printSummedPerformances = (problemCounter % printSummedStatsEveryNProblems == 0 )
        
        #if this was the last problem, print both stats:
        if (p == problems[-1]):
            printDecayingPerformances = True
            printSummedPerformances = True
            resultRecorder.saveRecords(filedir = './results/')
            
        if (printDecayingPerformances or printSummedPerformances):
            print("=== Round "+str(problemCounter)+" ===")
            print(selectorNames)
        if printDecayingPerformances:
            print("Decaying Ratios:")
            print("rel AUC       : "+str( decayingScores_auc / np.max(decayingScores_auc)) + " BEST: " + str( decayingScores_finalscore / np.max(decayingScores_finalscore) ) )
            print("AUC vs. Oracle: "+str( decayingScores_auc ) + " BEST: " + str( decayingScores_finalscore ) )
        if printSummedPerformances:
            print("Average Ratios:")
            print("rel AUC       : "+str( summedScores_auc / np.max(summedScores_auc)) + " BEST: " + str( summedScores_finalscore / np.max(summedScores_finalscore) ) )
            print("AUC vs. Oracle: "+str( summedScores_auc / problemCounter) + " BEST: " + str( summedScores_finalscore / problemCounter ) )
