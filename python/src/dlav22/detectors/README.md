# Detectos Format

## List of available Detectors
pose_detector (PoseDetector()): pif paf for pose detection, option to choose pose in a list of defined ones
yolo_detector (YoloDetector()): Detection using yolov5 can download the pretrained weights from online source
pose_yolo_detector (PoseYoloDetector()): running pifpaf for first detection and yolo otherwise, option to rerun pifpaf using start flag




## Format of the Detectors

Input: Original img
Output: bbox (x_center, y_center, w, h) or None
Class functions: all the detections bbox => list [x_center, y_center, width, height]

## To Do
config file for detectors: whch detector, which pose, wich weights for yolo
how it can be loaded (utils.deepsort)
Add more poses for pifpaf detector
Add option to change the weights used by yolo: s, m , l, x...
rename the detectors
top level =>think about how often we will restart pifpaf: time condition (every 20s), lost of track condition.... => will be implemented in top level file
