# Detectos Format

## List of available Detectors
base_detector (BaseDetector()): base class for detectors
pose_detector (PoseDetector()): pif paf for pose detection, option to choose pose in a list of defined ones
yolo_detector (YoloDetector()): Detection using yolov5 can download the pretrained weights from online source
color_detector (ColorDetector()): Detection based on color blobs.
pose_detector (PoseColorGuidedDetector()): PoseDetector module which uses an additional color filter if detection ambigous
pose_yolo_detector (PoseYoloDetector()): running pifpaf for first detection and yolo otherwise, option to rerun pifpaf using start flag


## Format of the Detectors

Input: Original img
Output: bbox (x_center, y_center, w, h) or None
Class functions: all the detections bbox => list [x_center, y_center, width, height]

## To Do
- [] config file for detectors: whch detector, which pose, wich weights for yolo
- [x] how it can be loaded (utils.deepsort)
- [] Add more poses for pifpaf detector
- [] Provide function for pifpaf to detect several people / bboxes -> and decide which one to use based on pose
- [x] Add option to change the weights used by yolo: s, m , l, x...
- [x]  rename the detectors
top level =>think about how often we will restart pifpaf: time condition (every 20s), lost of track condition.... => will be implemented in top level file
- [] Use logging instead of verbose
