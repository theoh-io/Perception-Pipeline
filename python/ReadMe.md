# **Robust Perception Module**

## Intro
This repository shows the implementation of a robust perception pipeline combining **Yolov5** Detection algorithm and **Deep Visual Re-Identification with Confidence** Tracking.

## User Guide


### QuickStart
Easiest way is to launch the bash script containing basic options for the algorithm
`./run.sh`  (make sure to have added execution permission with `sudo chmod +x`)



Other options are available: such as the choice of the detection algorithm and the tracker being used.

    python3 client.py --options

### Options

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