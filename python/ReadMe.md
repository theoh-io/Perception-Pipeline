# **Robust Perception Module**

## Intro
This repository shows the implementation of a robust perception pipeline combining **Yolov5** Detection algorithm and **Deep Visual Re-Identification with Confidence** Tracking.

## User Guide

### Connection with Loomo
make sure to be connected to the same wifi, get the ip adress of Loomo from settings. Then connect using `adb connect <ip Loomo>` and check connection is working using `adb devices`.

Run the AlgoApp on Loomo and press on button **Start Algo** to allow for socket connection, you should now see the camera on Loomo's screen.

### QuickStart
Easiest way is change the ip-adress of loomo in the bash script (changing each time) and launch the script containing basic options for the algorithm
`./run.sh`  (make sure to have added execution permission with `sudo chmod +x`)



Other options are available: such as the choice of the detection algorithm and the tracker being used.

    python3 client.py --options

### Options
`-i <ip-address Loomo>` used by socket to communicate with Loomo.

`-c <checkpoints>` path to the weights of the tracker. if no path is provided then no tracker is used.

`-d <downscale parameter>` impact the resolution of the video stream 2 by default and mus be changed accordingly in `script/follow.cfg` then pushed to Loomo using `adb push follow.cfg /sdcard/`

---
#### Options for detector
`-yt <float in [0;1]>` yolo-threshold value for confidence in detection.

---
#### Options for tracker
`--dist-metric <'L2', 'cosine'>` metric to compute distance between embeddings in the tracker

`-tt <tracker distance threshold>` distance threshold meaning that embedding refers to the same target as the reference (/!\ L2 distance must be below threshold, while cosine similarity must be above).

`ref-emb <method to get reference embedding: multiple, simple` multiple will keep fixed number of ref embedding and average across them while simple will only keep the last embedding.


## Structure

```
socket-loomo/python
│
│─── run.sh ───> bash script to launch algorithm with basic options
│
│─── client.py ───> main script to launch the algorithm
│
│─── detector.py ───> Definition of the class for each detector (Yolov5)
│
│─── tracker.py ───> Definition of the class for each tracker (Re-ID)
│
│─── test.py ───> simple script to test the video stream with Loomo
│
│─── ReID_model.pth.tar ───> weights of the Re-ID tracker trained on Market1501 dataset with confidence penalty
│
│─── yolov5s.pt ───> Pretrained weights for Yolov5s used for real-time detection
```

## Dependencies

## Citations