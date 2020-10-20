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
import json
### 3rd party imports (from packages, the environment) ### 
import numpy as np
### custom (local) imports ### 
from algs.base.evaluation import AlgorithmSelectionProcess, oracle, AlgorithmSelectionProcessRecorder

def run(selectors, selectorNames, algorithms, algorithm_names, problems, costs_matrix, scores_matrix, budgetfactor, resultPrefix = ''):
    summedScores_auc = np.zeros(len(selectors))
    summedScores_finalscore = np.zeros(len(selectors))
    decayingScores_auc = np.zeros(len(selectors))
    decayingScores_finalscore = np.zeros(len(selectors))
    decayFactor = 0.1
    regularization = 10^-5

    problemCounter = 0
    printDecayStatsAfterNProblems = 100
    printSummedStatsEveryNProblems = 100
    
    noOfSelectors = len(selectors)
    bestAlgorithms = np.zeros(len(algorithms)) #for counting how often one algorithm was the best one.
    
    resultRecorder = AlgorithmSelectionProcessRecorder(experiment_id = str(resultPrefix) + str(os.path.basename(__file__)), noOfAlgorithms = noOfSelectors, noOfProblems = len(problems), namesOfAlgorithms = selectorNames )
    
    for p in problems:
        problemCounter += 1
        def callback(algorithm):
            result = ( costs_matrix[algorithm, p], scores_matrix[algorithm, p] )
            return result
        
        def getMaxBudget():
            tCost = 0
            for a in algorithms:
                (cost, score) = callback(a)
                tCost += cost
            return tCost
            
        budget = getMaxBudget() * budgetfactor#max cost is 10, therefore 10*0.5 is about 50% of the budget required to go through half of the algorithms via random picking
        
        #print("=== Problem: "+str(p)+" b:"+str(budget)+" ===")
        
        (oCost, oScore, oAlgorithm, oSequence, oAUC) = oracle(algorithms, callback, budget)
        #oNormAUC = ( budget - oCost ) / budget
        bestAlgorithms[oAlgorithm] += 1
        #print("Oracle:   score "+str(oScore)+" algorithm: "+str(oAlgorithm)+" cost: "+str(oCost)+" NormAUC: "+str(oNormAUC))
        
        counter = 0
        for (selectorName, selector) in selectors:
            asprocess = AlgorithmSelectionProcess( budget = budget, callback = callback, measuretime = False )
            selector.process(budget = budget, callback = asprocess.callback)
            (bScore, bAlgorithm) = asprocess.getBest()
            nAUC = asprocess.normalizedAUC( maxAUCScore = oAUC )
            if (bScore is None):
                bScore = 0
            
            nBestScore = 0
            if ( oScore <= 0 ):
                oScore = regularization
                
            if not(oScore is None):
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
    
    sortedIndexes = np.argsort(bestAlgorithms)
    algorithms = np.flip((algorithms[sortedIndexes]))
    for alg_index in algorithms:
        if (bestAlgorithms[alg_index] > 0):
            print(str(algorithm_names[alg_index]) +" was "+str(int(bestAlgorithms[alg_index]))+" times the best one.")
            
    return resultRecorder


def benchmark_real_a(selectors, warmupfile = "./data/SKLearnOboeOnPregeneratedArtificialMLBenchmark_partial.json", validationfile = "./data/SKLearnOboeOnOpenMLBenchmark_partial.json", budgetfactor = 0.1, noOfWarmupRounds = 1, resultPrefix = ''):
    """
    This benchmark evaluates AlgorithmSelectors on two sets of historically calculated performance datasets.

    During the warmup-phase, the AS methods get to see the first set of instances and can use them to learn or warmstart their underlying system.
    During the validation-phase, the AS methods are measured and their performance compared with each other.
    Before the benchmarks are started, both provided files (warmup and validation) are investigated. Only the set of algorithms that appears in both is used for the evaluation.

    Parameters
    ----------
    selectors : list
        list of tuples, each consisting of a name and a reference to an AlgorithmSelector class.
        Note, the benchmark will later on create instances of the algorithm selector classes and set the number of algorithms.
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

    with open(warmupfile, 'r') as infile:
        warmup_data = json.load(infile)
    

    with open(validationfile, 'r') as infile:
        validation_data = json.load(infile)
    
    #assert(np.all( warmup_data['algorithms'] == validation_data['algorithms'] )), 'Different algorithms in the warmup and validation set detected. This prevents further execution of the benchmark.'

    algorithm_names_warmup = warmup_data['algorithms']
    algorithm_names_validation = validation_data['algorithms']
    
    #print("Algorithm names warmup:")
    #print(algorithm_names_warmup)
    
    #print("Algorithm names validation:")
    #print(algorithm_names_validation)
    
    algorithm_names = np.intersect1d(algorithm_names_warmup, algorithm_names_validation)
    alg_n = len(algorithm_names)
    algorithm_indexes = np.arange(alg_n)
    
    print("Benchmark Real A")
    print("no of algorithms: "+str(alg_n))
    print(algorithm_names)
    
    #for extracting only the columns of the score and cost matrices that are from the common algorithms present in both:
    alg_indexes_warmup = list()
    alg_indexes_validation = list()
    
    for a in algorithm_names:
        for i in range(len(algorithm_names_warmup)):
            if a == algorithm_names_warmup[i]:
                alg_indexes_warmup.append(i)
                break
        for i in range(len(algorithm_names_validation)):
            if a == algorithm_names_validation[i]:
                alg_indexes_validation.append(i)
                break

    #print("Where: "+str(np.where( algorithm_names == algorithm_names_warmup[0])[0][0]))
    alg_indexes_warmup = np.array(alg_indexes_warmup)
    alg_indexes_validation = np.array(alg_indexes_validation)
    #alg_indexes_warmup = np.array([algorithm_names.index(a) for a in algorithm_names_warmup])
    #alg_indexes_validation = np.array([algorithm_names.index(a) for a in algorithm_names_validation])


    #initialize the selector instances from the provided classes
    selectors2 = list()
    selectorNames = list()
    for s in selectors:
        selectors2.append((s[0],s[1](algorithms = algorithm_indexes)))
        selectorNames.append(s[0])
    selectors = selectors2

    #Run warmup phase
    
    print("####################")
    print("### Warmup Phase ###")
    print("####################")
    costs = np.array( warmup_data['costs'] )[alg_indexes_warmup,:]
    scores = np.array( warmup_data['scores'] )[alg_indexes_warmup,:]
    for i in range(noOfWarmupRounds):
        resultRecorder = run(selectors, selectorNames, algorithm_indexes, algorithm_names, np.arange(len(warmup_data['tasks'])), costs, scores, budgetfactor, resultPrefix = resultPrefix + '_warmup_'+str(i))
        resultRecorder.setMetadata('warmupfile', str(warmupfile))
        resultRecorder.setMetadata('validationfile', str(validationfile))
        resultRecorder.setMetadata('budgetfactor', str(budgetfactor))
        resultRecorder.saveRecords(filedir = './results/')
    
    
    #Run validation phase
    print("########################")
    print("### Validation Phase ###")
    print("########################")
    costs = np.array( validation_data['costs'] )[alg_indexes_validation,:]
    scores = np.array( validation_data['scores'] )[alg_indexes_validation,:]
    resultRecorder = run(selectors, selectorNames, algorithm_indexes, algorithm_names, np.arange(len(validation_data['tasks'])), costs, scores, budgetfactor, resultPrefix = resultPrefix)
    resultRecorder.setMetadata('noOfWarmupRounds', str(noOfWarmupRounds))
    resultRecorder.setMetadata('warmupfile', str(warmupfile))
    resultRecorder.setMetadata('validationfile', str(validationfile))
    resultRecorder.setMetadata('budgetfactor', str(budgetfactor))
    resultRecorder.saveRecords(filedir = './results/')
