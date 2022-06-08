#!/bin/bash

# #Benchmarking only stark on all Loomo videos
# path_gt="Benchmark/LoomoBenchmark2"
# path_det="Results/stark"

END=16
for i in $(seq 1 $END)
    do
    name_file="Loomo${i}"
    echo "name of file: ${name_file}"
    python3 modif_cfg.py -id $i -v
    python3 run.py 
    done