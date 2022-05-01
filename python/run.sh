#!/bin/bash

python3 client.py -c ReID_model.pth.tar -i 128.179.180.5 -v 'True' -ym 'yolov5n' -yt 0.4 --dist-metric 'cosine' -tt 0.6 --ref-emb 'simple'
