#!/bin/bash

#Unitary test
# path_gt="../../Benchmark/LoomoBenchmark2/Loomo006_gt.txt"
# path_det="../../Results/stark/ID_0006_prediction.txt"
# python3 benchmark.py -gt $path_gt -det $path_det 

# #Test with Averaging on multiple videos
# path_gt="../../Benchmark/LoomoBenchmark2"
# path_det="../../Results/deepsort"
path_gt="../../Benchmark/Skater/groundtruth_rect.txt"
path_det="../../Results/OTB100/stark/ID_0002_prediction.txt"
python3 benchmark.py -gt $path_gt -det $path_det -s -v -x1y1


# #Benchmarking only stark on all Loomo videos
# path_gt="Benchmark/LoomoBenchmark2"
# path_det="Results/stark"

# END=16
# for i in $(seq 1 $END)
#     do
#     name_file="Loomo${i}"
#     echo "name of file: ${name_file}"
#     path_detect="${path_det}/${name_file}_det.txt"
#     path_groundtruth="${path_gt}/${name_file}_gt.txt"
#     python3 get_metrics.py  -gt $path_groundtruth -det $path_detect -iou 0.5
#     done

