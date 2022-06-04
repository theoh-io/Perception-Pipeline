#!/bin/bash

cd ../src/dlav22/deep_sort/deep/checkpoint/

#downloading the models and renaming to fit with deepsort convention
FILE="resnet50_AGWmarket.pth"
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
fi
if [ ! -f "$FILE" ]; then
    wget https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth
    mv market_agw_R50.pth resnet50_AGWmarket.pth
fi

FILE="resnet50_SBSmsmt17.pth"
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
fi
if [ ! -f "$FILE" ]; then
    wget https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50.pth
    mv msmt_sbs_R50.pth resnet50_SBSmsmt17.pth
fi

FILE="resnet50_SBSduke.pth"
if [ -f "$FILE" ]; then
    echo "$FILE already exists"
fi
if [ ! -f "$FILE" ]; then
    wget https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50.pth
    mv duke_sbs_R50.pth resnet50_SBSduke.pth

fi




