#!/bin/bash

# #Test with Averaging on multiple videos
path_gt="../../Benchmark/LoomoBenchmark2"
path_det="../../Results/deepsort"
python3 benchmark.py -gt $path_gt -det $path_det -v -x1y1



