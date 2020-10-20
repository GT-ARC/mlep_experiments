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

### 3rd party imports (from packages, the environment) ### 
import numpy as np
### custom (local) imports ### 
from benchmark_synthetic_a import benchmark_synthetic_a
from benchmark_synthetic_b import benchmark_synthetic_b
from benchmark_synthetic_c import benchmark_synthetic_c
from benchmark_synthetic_d import benchmark_synthetic_d
from benchmark_synthetic_d2 import benchmark_synthetic_d2
from benchmark_real_a import benchmark_real_a
from algs.mc import MonteCarloAlgorithmSelector
from algs.cas import ConservativeAlgorithmSelector
from algs.ars import AverageRankingSelector
#from algs.rrs import NemesisSelector
from algs.knn import KNeighborsSelector

#from algs.aas import AdvancedAlgorithmSelector
#from algs.nnas import NeuralNetworkAlgorithmSelector
from algs.nnasauc import NeuralNetworkAlgorithmSelectorAUC
#from algs.nnasrev2 import NeuralNetworkAlgorithmSelectorR2
from algs.nnasrev3 import NeuralNetworkAlgorithmSelectorR3
from algs.ebas import EnsembleBasedAlgorithmSelector
from algs.bo import BayesianOptimizationAlgorithmSelector


#Create the algorithm selector
selectors = list()
selectors.append( ("MonteCarlo", MonteCarloAlgorithmSelector) ) 
selectors.append( ("Average Ranking", AverageRankingSelector) )  
selectors.append( ("Conservative", ConservativeAlgorithmSelector) )  
selectors.append( ("Bayesian", BayesianOptimizationAlgorithmSelector) )  
#selectors.append( ("Nemesis", NemesisSelector) )  
#selectors.append( ("Advanced", AdvancedAlgorithmSelector) )   
#selectors.append( ("NNAS", NeuralNetworkAlgorithmSelector) ) 
#selectors.append( ("NNASAUC", NeuralNetworkAlgorithmSelectorAUC) ) 
#selectors.append( ("NNASR2", NeuralNetworkAlgorithmSelectorR2) )
#selectors.append( ("KNeighbours", KNeighborsSelector) ) 
#selectors.append( ("NNASR3", NeuralNetworkAlgorithmSelectorR3) ) 
#selectors.append( ("EBAS", EnsembleBasedAlgorithmSelector) ) 
benchmark_synthetic_a( selectors = selectors, alg_n = 100, problem_n = 100, budgetfactor = 0.25 )
#benchmark_synthetic_b( selectors = selectors, alg_n = 100, problem_n = 1000, budgetfactor = 0.1 )
#benchmark_synthetic_c( selectors = selectors, alg_n = 100, problem_n = 100000, budgetfactor = 0.1 ) #additionally simulating "always bad" algorithms.
#benchmark_synthetic_d( select#ors = selectors, problem_n = 100000 )#this is more of a "debug" benchmark. It contains 3 algorithms, one to determine which of the other both should be run to get max score.
#benchmark_synthetic_d2( selectors = selectors, alg_n_half = 100, problem_n = 100000 )#this is more of a "debug" benchmark. It contains 3 algorithms, one to determine which of the other both should be run to get max score.
#benchmark_real_a( selectors = selectors, warmupfile = "./data/SKLearnOboeOnPregeneratedArtificialMLBenchmark_partial.json", validationfile = "./data/SKLearnOboeOnOpenMLBenchmark_partial.json", budgetfactor = 0.1)

#benchmark_real_a( selectors = selectors, warmupfile = "./data/SKLearnOboeOnPregeneratedArtificialMLBenchmark_partial.json", validationfile = "./data/SKLearnOboeOnPennMLBenchmark_partial.json", budgetfactor = 0.1, noOfWarmupRounds = 0, resultPrefix = 'coldstart_pennml')

#benchmark_real_a( selectors = selectors, warmupfile = "./data/SKLearnOboeOnPregeneratedArtificialMLBenchmark_partial.json", validationfile = "./data/SKLearnOboeOnPennMLBenchmark_partial.json", budgetfactor = 0.1, noOfWarmupRounds = 1, resultPrefix = 'warmstart1x_pennml')