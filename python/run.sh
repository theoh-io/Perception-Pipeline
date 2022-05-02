#!/bin/bash

python3 client.py -c ReID_model.pth.tar -i 128.179.182.176 -v true -ym 'yolov5s' -yt 0.4 --dist-metric 'cosine' -tt 0.6 --ref-emb 'multiple' --nb-ref 20 --av-method linear --intra-dist 5


