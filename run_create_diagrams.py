'''
ASF-BLATS
Algorithm Selection Framework
for budget limited any-time scenarios

This file contains scripts to aggregate the resulting data from the experiments (run_benchmarks.py) and generate diagrams from them.

Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2019 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
'''

### python basic imports ###
import argparse
import os
import json
import re
from pathlib import Path
### 3rd party imports (from packages, the environment) ### 
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
### custom (local) imports ### 



RESULTPATH = './results/'

'''
Moving average from:https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
'''
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
    
def running_median(x, N):
    return medfilt(x,N)
    #cumsum = np.cumsum(np.insert(x, 0, 0)) 
    #return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='aggregate and plot run_benchmark experiment results')
    
    
    possibleFiles = []
    resultfilepattern = re.compile(pattern = '^.+\.json$', flags=0)
    genericEndingPattern = re.compile(pattern = '_[0-9]+\.json$', flags=0)
    resultDictionary = dict()
    
    for filename in os.listdir(RESULTPATH):
        match = resultfilepattern.search(filename)
        if ( match ):
            possibleFiles.append(filename)

            endingMatch = genericEndingPattern.search(filename)
            
            if endingMatch:
                print()
                strippedFileName = filename[:-len(endingMatch.group(0))]
            
            if strippedFileName in resultDictionary:
                resultDictionary[strippedFileName] = resultDictionary[strippedFileName]  + 1
            else:
                resultDictionary[strippedFileName] = 1
    
    parser.add_argument('--input','-i',
                dest='inputfiles',
                action='store',
                type=type(' '),
                nargs='+',
                required=False,
                help='the resulting .json file after running run_benchmarks.py. Typically located in the '+str(RESULTPATH)+' folder. Example options: '+str(possibleFiles[:5]))
                
    parser.add_argument('--inputBatch','-b',
                dest='inputBatchFile',
                action='store',
                type=type(' '),
                nargs='+',
                required=False,
                default=possibleFiles,
                help='similar to -i but assumes that there are similar files in the pattern <name>_0.json in the '+str(RESULTPATH)+' folder that should all be considered. Options: '+str(resultDictionary))

    parser.add_argument('--slidingWindowSize','-s',
                    dest='slidingWindowSize',
                    action='store',
                    type=type(' '),
                    default='1',
                    help='number of data points that are averaged over to smoothen the curve.')
                    
    parser.add_argument('--concat','-c',
            dest='concatenate',
            action='store_true',
            default='False',
            help='concatenate the series of data to form a long sequence in one diagram instead of seperate diagrams.')
            
    parser.add_argument('--average','-a',
            dest='average',
            action='store_true',
            default='False',
            help='average multiple provided diagrams (need to be the same length, otherwise an error will be raided)')
            
    parser.add_argument('--median','-m',
            dest='median',
            action='store_true',
            default='False',
            help='calculates the median instead of the mean within the sliding window.')
                    
    args = parser.parse_args()
    
    inputFiles = []
    for genericInputFile in args.inputBatchFile:
        i = 0
        filename = genericInputFile+'_'+str(i)+'.json'
        filepath = Path(RESULTPATH+filename)
        while ( filepath.exists() and os.path.isfile(filepath) ):
            inputFiles.append(filepath)
            i += 1
            filename = genericInputFile+'_'+str(i)+'.json'
            filepath = Path(RESULTPATH+filename)
    
    if not ( args.inputfiles == None ):
        for inputfile in args.inputfiles:
            filepath = Path(RESULTPATH+inputfile)
            if filepath.exists() and os.path.isfile(filepath):
                inputFiles.append(filepath)
    print("Input files:")
    print(inputFiles)
    
    print("Concatenate: "+str(args.concatenate))
    print("Average: "+str(args.average))
    aggregatedData = dict()

    #Open the first file
    with open(inputFiles[0], 'r') as infile:
        reference_root = json.load(infile)
        reference_metadata = reference_root[0]
        reference_data = reference_root[1]
        
        reference_experimentId = reference_metadata['experiment_id']
        reference_noOfAlgorithms = int( reference_metadata['noOfAlgorithms'] )
        reference_noOfProblems = int( reference_metadata['noOfProblems'] )
        reference_namesOfAlgorithms = reference_metadata['namesOfAlgorithms']
        
        print(reference_metadata)
        
    for filepath in inputFiles:
        with open(filepath, 'r') as infile:
            root = json.load(infile)
            metadata = root[0]
            data = root[1]
            
            experimentId = metadata['experiment_id']
            noOfAlgorithms = int( metadata['noOfAlgorithms'] )
            noOfProblems = int( metadata['noOfProblems'] )
            namesOfAlgorithms = metadata['namesOfAlgorithms']
            
            print(metadata)
            
            assert ( reference_noOfAlgorithms == noOfAlgorithms ), 'no of algorithms unequal to the reference'
            assert ( reference_noOfProblems == noOfProblems ), 'no of problems unequal to the reference'
            
            for (key, data) in data.items():
                if (key in aggregatedData):
                    if ( args.concatenate == True ):
                        aggregatedData[key] = np.concatenate( (aggregatedData[key], data), axis = 1 )
                    elif ( args.average == True ):
                        aggregatedData[key] = np.add( aggregatedData[key], data, aggregatedData[key] )
                else:
                    aggregatedData[key] = np.array(data)

    if ( args.average == True ):
        print("args.average: "+str(args.average))
        avgFactor = 1.0 / len(inputFiles)
        for key in aggregatedData.keys():
            aggregatedData[key] = aggregatedData[key] * avgFactor

    if not( args.concatenate == True or args.average == True ):
        aggregatedData = reference_data
        #TODO: Make sure that all files are plotted, not only the first one (for single file input mode, batch mode is not affected).
            
        

    data = aggregatedData
    
    experimentId = reference_experimentId
    noOfAlgorithms = reference_noOfAlgorithms
    namesOfAlgorithms = reference_namesOfAlgorithms
    
    #print("available data: ")
    #print(data.keys())
    #naucData = np.asarray(data['nauc'])
    #nBestScore = np.asarray(data['nBestScore'])
    
    figsize = (16,12)

    
    for (key, data) in data.items():
        data = np.array(data)

        #calculate rolling mean over N points for each algorithm:
        N = int(args.slidingWindowSize)
        if (N > 1):
            for i in range(noOfAlgorithms):
                if (args.median == True):
                    N2 = int(N*0.5)
                    data[i,:-N+1] = running_median(data[i,:],N)[N2:-N2]
                else:
                    data[i,:-N+1] = running_mean(data[i,:],N)
            data = data[:,:-N+1]
        
        plt.figure(figsize=figsize)
        plt.plot(data.T)
        plt.legend(namesOfAlgorithms)
        plt.title(experimentId + ' '+key)
        #plt.savefig(filepath + 'pca_scores.png')
        plt.show(block=True)
        #plt.pause(diagram_display_time)
        plt.close()
