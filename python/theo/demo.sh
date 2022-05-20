#!/bin/bash

python3 client.py -d 2 -c ReID_model.pth.tar -i 128.179.189.44 -v true -ym 'yolov5s' -yt 0.15  --dist-metric 'cosine' -tt 0.87  --ref-emb 'multiple' --nb-ref 20 --av-method linear  --recording '../recording/Demo/Demo8.avi'


