# Algorithm Selection Strategies Documentation #
In this document, you will find the technical documentation of various algorithm selection methods and their implementations. We will start with the problem description and problem interface. After that introduction, for each of the different implementations, we will give detailed description about the methods and implementation. Since all implementations share a common interface, they can be easily swapped out for each other. This allows for numerical evaluating them on various artificial and realistic use cases and make them easy to use in a real application as well.

# Problem Interface #
In CODA, we consider the problem of algorithm selection in a budget-constrained anytime-performance environment. Consider a problem instance p_i that various algorithms a_0, a_1, a_2 element of A can be applied on. Applying an algorithm on a problem instance costs budget and yields information about the performance (cost and score) of this algorithm on the specific current problem instance. But this only describes the inner problem setting. The global problem setting considers not just one problem instance, but a sequence of problem instances, each with it's own budget and algorithm performances. In order to exploit this additional outer setting, we may assume that the algorithm performances are not completely independend from each other, e.g. the performance of one algorithm on one problem instance might be used to predict the performances of other algorithms on the same problem instance, given historical information about thoose relationships on previously seen problem instances.

The common python interface we designed reflects this scenario. First, when initializing the algorothm selection method, we need to define the set of algorithms it should select from.

See /algs/base/framework.py
class AlgorithmSelector
def __init__(self, algorithms)

where algorithms is the set of algorithms (may be a set of id's, numbers or even object instances). To start the actual algorithm selection process, we call the function 

def process(self, budget, callback)

providing it with a cost budget (a float) and a problem instance specific callback function that takes one of the elements in the set of algorithms and returns the score or a ( cost, score ) tuple of that algorithm run on the problem instance. Note: If it only returns a score, the runtime costs will be measured by another components as the timethat callack function actually requires to finish. For artificial setups with precalculated costs, we added the function of explicitly returning the costs to speed things up.

Note: each call to process() represents another problem instance, so the different algorithm selection approaches need to consider this when building their meta-features repositories internally.


## Baseline MC ##
As a simple baseline, we implemented a monte-carlo (mc) algorithm selection method in mc.py. This method picks a random algorithm from the set without redrawing.

## Baseline Average Ranking ##
As another baseline, we implemented an average ranking method that just decides on the average costs and scores over historically seen runs. It greedily picks the one with the best score/cost ratio first.

## Learning to Rank #
In this method, we fit linear regression models via stochastic gradient decent to predict scores and costs of algorithms based on the current information state. The information state consists of a vector that is the concatenation of so far observerd scores, costs and a binary indicator about if an algorithm has been run yet. See aas.py class AdvancedAlgorithmSelector.

# Recommender Systems #
This simple approach normalizes all the tested algorithms from one run and keeps average performances. Whenever a new problem instance occurs, it randomly (with increasing chance) recommends the algorithm with the best score/runtime ratio (exploitation). Otherwise, it will choose a random algorithm to extend his world-view (exploration). See cas.py for the implementation.

# kNN #
The knn-based method predicts the runtime based on the k-closest previously seen algorithm runs. "Close" is defined by the current known information vector. This method generally quickly outperforms the baseline method, but stagnated at a certain point, because it does not learn a good strategy for the very beginning (when no information is known) and suffers from an increasing model size (e.g. the historical runs that are recorded).
See knn.py for the code.

# Multi-Armed Bandit - Reinforcement Learning #
This method applies reinforcement learning on the problem. Although in theory, this approach should work well, the lack of reproducability in this setting (no state is exactly reproducible) creates a challenge. See nnasrev3.py for the implementation and further information.

# License and Acknowledgement #
Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2020 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.