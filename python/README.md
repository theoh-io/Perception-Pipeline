# Modular Pipeline Configurations & User Guide

## Structure perceptionloomo package

    ├── scripts     # run inference on Benchmark or Loomo and others...
    ├── src/perceptionloomo     # Folder for the package perceptionloomo
        ├---- configs           # Config files to customize the pipeline
        ├---- detectors         # Code for Yolo and PifPaf detectors
        ├---- trackers          # Code for the Trackers
        ├---- perception        # Code for the High level Detector class
        ├---- deep_sort         # Modified code implementing deepsort 
        ├---- mmtracking        # ressources used for siam and stark 


## Options Available

### Detection
To customize the Detection module modify cfg_detector.yaml

- Detection Module:
    - Yolov5 Only (YoloDetector)
    - Initialization with Pifpaf and then Yolo (PoseYoloDetector)
- Type of Yolov5 architecture (s, m, l, x)
- Pose being used by Openpifpaf

### Tracking
To customize the Detection module modify cfg_trackers.yaml

- Tracker Module:
    - Custom ReID Tracker (ReIdTracker)
    - Deepsort (DSCustomReid)
    - Stark (SotaTracker)
    - SiamRPN++ (SotaTracker)

- Deepsort:
    - ReID model to use: by default osnet_x0_25 (a lot already implemented and proposed for download)
    - Other Deepsort parameters: max_dist, max_iou, max_age, n_init, nn_budget

- ReID Tracker:
    - Weights to use: default trained on market1501 with confidence loss penalty (see [here](https://github.com/vita-epfl/Deep-Visual-Re-Identification-with-Confidence))
    - distance metric: L2 or cosine
    - Similarity threshold: above/under (cosine/L2) the defined distance 2 embeddings should represent same person

- MMTRACKING:
    - Device: default None to use GPU if available, can specify cpu
    - Config: path to the config file for the specific model (configs can be found in src/perceptionloomo/mmtracking/configs)
    - Model: Downloaded Models (siamese and stark)
    - Conf: Confidence threshold, below this number tracking will not be taken into account





