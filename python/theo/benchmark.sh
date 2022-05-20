#!/bin/bash


name_file="Human9"
python3 inference.py -s "../../Benchmark/${name_file}" -c ReID_model.pth.tar -ym 'yolov5s' -yt 0  --dist-metric 'cosine' -tt 0.87  --ref-emb 'multiple' --nb-ref 15 --av-method "linear" \
        --ground-truth "../../Benchmark/${name_file}/groundtruth_rect.txt" --init-det 1 \
        --recordingbb "../../Results/${name_file}/inference.avi"

#script for loomo: no ground truth provided
# name_file="Loomo"
# python3 inference.py -s "../Benchmark/${name_file}" -c ReID_model.pth.tar -ym 'yolov5s' -yt 0  --dist-metric 'cosine' -tt 0.9  --ref-emb 'multiple' --nb-ref 15 --av-method "linear" \
#         --init-det 1 \
#         --recordingbb "../Results/${name_file}/inference.avi"

