

from dlav22.perception import DetectorG16
import dlav22
from dlav22.utils.utils import Utils
from dlav22 import detectors


det = DetectorG16()
print(det.cfg)
print(det.cfg.PERCEPTION.VERBOSE)
print(det.cfg.TRACKER.TRACKER_CLASS)

Utils.import_from_string(det.cfg.TRACKER.TRACKER_CLASS)(det.cfg) 

from importlib import import_module

class_str: str = 'dlav22.detectors.pose_yolo_detector.PoseYoloDetector'

try:
    module_path, class_name = class_str.rsplit('.', 1)
    module = import_module(module_path)
    obj = getattr(module, class_name)()
except (ImportError, AttributeError) as e:
    raise ImportError(class_str)

print(obj.start)