#!/bin/bash

#Need to be launched from python using ./sheelscript/inference_vid.sh
name_file="Loomo/Demo1"
name_file="Human3"
python3 inference.py  -c models/ReID_model.pth.tar -s "../../Benchmark/${name_file}" -v -ym 'yolov5s' -yt 0  --dist-metric 'cosine' -tt 0.85  --ref-emb 'smart' --av-method 'standard' --nb-ref 10 #--recordingbb "Output/LoomoVideo/${name_file}/smart_85_10.avi" 
