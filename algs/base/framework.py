"""
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains the basic classes for the whole framework.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
"""

### python basic imports ###

### 3rd party imports (from packages, the environment) ### 

### custom (local) imports ### 

### class definitions ###
class AlgorithmSelector:
    """
    Main basic interface for algorithm selection implementations. Note: This objects life time is longer than a single algorithm selection process on a single problem instance. It actually allows to keep and exploit meta-knowledge between multiple algorithm selection processes. In the future, and extention to save and load the internal state of an algorithm selection is planned.
    
    Intended usage
    --------------
    Inherit this interface and implement the __init__ and process methods as needed.
    """
    def __init__(self, algorithms):
        """
        Parameters
        ----------
        algorithms : COLLECTION
            a set, array or other collection holding objects that represent the algorithm to selection from. This are the objects or ids passed to the callback function in process.
        """
        return NotImplemented
        
    def process(self, budget, callback):
        """
        Start an algorithm selection process. This represents a single algorithm selection process on one problem instance. During this process, multiple calls to the callback function can be made to evaluate the different algorithms.
        
        Parameters
        ----------
        budget : float
            time budget for the whole algorithm selection process.
        callback : FUNCTION
            function that takes a single algorithm from the algorithms collection as input and returns a score : float that indicates how well that algorithm did.
        """
        return NotImplemented


class OutOfTimeException(Exception):
    """
    A custom exception class that indicates we run out of time.
    """
    pass