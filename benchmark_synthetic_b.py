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
from algs.base.evaluation import AlgorithmSelectionProcess, oracle, AlgorithmSelectionProcessRecorder



def benchmark_synthetic_b(selectors, alg_n = 100, problem_n = 10000, budgetfactor = 0.25):
    """
    This synthetic benchmark evaluates AlgorithmSelectors on a bunch of generical algorithms and problems.

    The algorithms are designed with the following properties:
    The score is normed to be between (1,10].
    The runtime is normed to be between [1,10].
    There is no guarantee that any algorithm on a given problem achieves the maximum or minimum score.
    We simulate "hard" problems that require algorithms with higher cost.
    We simulate relations between algorithms, meaning that one can learn to map performances from  cheaper to well working more expensive algorithms.

    Parameters
    ----------
    selectors : list
        list of tuples, each consisting of a name and a reference to an AlgorithmSelector class.
        Note, the benchmark will later on create instances of the algorithm selector classes and set the number of algorithms.
    alg_n : integer
        number of synthetic algorithms created and used in the benchmark.
    problem_n : integer
        number of problems to evaluate on. Should be much larger than the number of algorithms.
    budgetfactor : float
        how much time with respect to the overall required time to run all algorithms is provided (per problem)

    Returns
    -------
    float
        The average Area under the curve with respect to the highest observed AUC

    Example usage
    -------------
    selectors = list()
    selectors.append( ("MonteCarlo", MonteCarloAlgorithmSelector) )
    selectors.append( ("Conservative", ConservativeAlgorithmSelector) )   
    benchmark_synthetic_a(selectors = selectors, alg_n = 100, problem_n = 100000 )
    """
    algorithms = np.arange(0,alg_n) #x algorithms
    np.random.shuffle(algorithms) #shuffle the algorithms in place to destroy their smoothness
    problems = np.random.rand(problem_n)*len(algorithms) #random vector of size=problem_n with numbers between 0 and alg_n.
    
    #initialize the selector instances from the provided classes
    selectors2 = list()
    selectorNames = list()
    for s in selectors:
        selectors2.append((s[0],s[1](algorithms = algorithms)))
        selectorNames.append(s[0])
    selectors = selectors2
    
    resultRecorder = AlgorithmSelectionProcessRecorder(experiment_id = os.path.basename(__file__), noOfAlgorithms = alg_n, noOfProblems = len(problems), namesOfAlgorithms = selectorNames )
    summedScores_auc = np.zeros(len(selectors))
    summedScores_finalscore = np.zeros(len(selectors))
    decayingScores_auc = np.zeros(len(selectors))
    decayingScores_finalscore = np.zeros(len(selectors))
    decayFactor = 0.1

    problemCounter = 0
    printDecayStatsAfterNProblems = 10
    printSummedStatsEveryNProblems = 10
    
    
    
    for p in problems:
        problemCounter += 1
        def callback(algorithm):
            simulatedCost = np.power( 10, (p / alg_n) * (algorithm % 100)/100 ) #range: (1, 10), exponentially distributed. Note that some problems are "harder" and require higher costs.
            simulatedScore = 10 * ( 1.0 - ( np.abs(p - algorithm) / alg_n ) )# 1-10, based on their distance, favoring further away algorithms
            return ( simulatedCost, simulatedScore )
            
        
        def getMaxBudget():
            tCost = 0
            for a in algorithms:
                (cost, score) = callback(a)
                tCost += cost
            return tCost
            
        budget = getMaxBudget() * budgetfactor#max cost is 10, therefore 10*0.5 is about 50% of the budget required to go through half of the algorithms via random picking
        
        #print("=== Problem: "+str(p)+" b:"+str(budget)+" ===")
        
        (oCost, oScore, oAlgorithm, oSequence, oAUC) = oracle(algorithms, callback, budget)


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