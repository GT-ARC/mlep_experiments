# Algorithm Selection Framework for budget limited any-time scenarios
This repository contains the implementations and benchmarks for meta learning evaluation policies for selecting machine learning algorithms in a budget-constrained setting.

For the algorithms, see the [subfolder /algs](/algs/readme.md).

# Benchmarks
Running a benchmark happens by modifying the script run_benchmarks.py and then executing it with python. Within that file, one can comment/uncomment which mlep-approaches should be tested and on which benchmark they should be run. The individual benchmarks are setup in the files benchmarks_*.py.

The benchmark_real_a.py takes precalculated performance data and runs the benchmark on thoose. We have included performance data collected from our GPU-Clusters in /data for the PennML Benchmark dataset, the OpenML Dataset and our own artificial datasets. Since one can define the warmup dataset independently from the evaluation dataset, most of the benchmarks use the artificial data for warmstarting the learning and then using the real datasets from pennML or openml for evaluation to maximize the amount of real datasets used for evaluation. Of course, depending on the meta-learning approach, algorithms can suffer from this because they try to estimate the distribution of the performance data instead of learning a good overall strategy. Therefore, a separate experiment without warmstart data is advised to see such differences.

# Plotting results
For visualization of the benchmark results, one can use run_create_diagrams, which plots several different aspects of the benchmark.

# License and Acknowledgements
* Author: Christian Geißler <christian.geissler@gt-arc.com>
* License: Copyright 2019 by GT-ARC gGmbH
* Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.
