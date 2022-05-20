#!/bin/bash

python3 client.py -d 4 -c ReID_model.pth.tar -s 'loomo' -i 128.179.187.124 -v true -ym 'yolov5s' -yt 0  --dist-metric 'cosine' -tt 0.91  --ref-emb 'smart' --nb-ref 20 --av-method exp --intra-dist 5 --recording '../../recording/loomo3.avi'


