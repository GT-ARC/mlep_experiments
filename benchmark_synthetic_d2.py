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



def benchmark_synthetic_d2(selectors, alg_n_half = 5, problem_n = 1000):
    """
    This synthetic benchmark evaluates AlgorithmSelectors on a very specificly crafted task.
    The benchmark measures how well the AS method can exploit a clear logical relationship.
    
    The set of 2 * alg_n algorithms has the following properties:
    
    BEST = random( alg_n )
    score(algorithm_a0) = score(algorithm_b0) * 0.0001
    cost(algorithm_a0) = 0.1 / alg_n
    cost(algorithm_b0) = 1.0
    budget = 1.2
    
    score(algorithm_bBEST) = 1.0
    score(algorithm_b!=BEST) = 0.0

    Parameters
    ----------
    selectors : list
        list of tuples, each consisting of a name and a reference to an AlgorithmSelector class.
        Note, the benchmark will later on create instances of the algorithm selector classes and set the number of algorithms.
    problem_n : integer
        number of problems to evaluate on. Should be much larger than the number of algorithms.

    Returns
    -------
    float
        The average Area under the curve with respect to the highest observed AUC

    Example usage
    -------------
    selectors = list()
    selectors.append( ("MonteCarlo", MonteCarloAlgorithmSelector) )
    selectors.append( ("Conservative", ConservativeAlgorithmSelector) )   
    benchmark_synthetic_a(selectors = selectors, problem_n = 100000 )
    """
    algorithms = np.arange(alg_n_half * 2)
    problems = np.random.rand(problem_n) #random vector of size=problem_n with numbers between 0 and alg_n.
    
    #initialize the selector instances from the provided classes
    selectors2 = list()
    selectorNames = list()
    for s in selectors:
        selectors2.append((s[0],s[1](algorithms = algorithms)))
        selectorNames.append(s[0])
    selectors = selectors2
    
    resultRecorder = AlgorithmSelectionProcessRecorder(experiment_id = os.path.basename(__file__), noOfAlgorithms = len(selectors), noOfProblems = len(problems), namesOfAlgorithms = selectorNames )
    summedScores_auc = np.zeros(len(selectors))
    summedScores_finalscore = np.zeros(len(selectors))
    decayingScores_auc = np.zeros(len(selectors))
    decayingScores_finalscore = np.zeros(len(selectors))
    decayFactor = 0.1

    problemCounter = 0
    printDecayStatsAfterNProblems = 100
    printSummedStatsEveryNProblems = 100
    
    for p in problems:
        problemCounter += 1
        ground_truth = np.random.randint(0,alg_n_half)
        def callback(algorithm):
            if (algorithm < alg_n_half):
                #algorithms 0 till alg_n_half-1 are the probing algorithms
                
                cost = 0.1 / alg_n_half
                if (algorithm == ground_truth):
                    return cost, 0.1
                else:
                    return cost, 0
            else:
                #algorithms alg_n till 2*alg_n-1 are the actual scoring algorithms
                cost = 1.0
                if ((algorithm-alg_n_half) == ground_truth):
                    return cost, 1.0
                else:
                    return cost, 0.0


            
        budget = 2.9 # 1.0 for the "best one", 0.1 for probing and 0.8 to counter small inaccuracies and getting enough reward from the best choosen one. Note: 1.9 budget still does not allow to run two bigger scoring algorithms.
        
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