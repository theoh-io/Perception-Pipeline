#!/usr/bin/env python

from dlav22.detectors.pose_detectors  import PifPafDetector, PifPafKeyPoints
import numpy as np

def test_check_if_desired_pose():

    test_key_points = np.ones((16,3))
    test_key_points[PifPafKeyPoints.LEFT_WRIST.value,0] = 1
    test_key_points[PifPafKeyPoints.LEFT_WRIST.value,1] = 0

    test_key_points[PifPafKeyPoints.RIGHT_WRIST.value,0] = 3
    test_key_points[PifPafKeyPoints.RIGHT_WRIST.value,1] = 0

    test_key_points[PifPafKeyPoints.LEFT_SHOULDER.value,0] = 1
    test_key_points[PifPafKeyPoints.LEFT_SHOULDER.value,1] = 5

    test_key_points[PifPafKeyPoints.RIGHT_SHOULDER.value,0] = 3
    test_key_points[PifPafKeyPoints.RIGHT_SHOULDER.value,1] = 5

    test_key_points[PifPafKeyPoints.LEFT_ELBOW.value,0] = 4
    test_key_points[PifPafKeyPoints.LEFT_ELBOW.value,1] = 2

    test_key_points[PifPafKeyPoints.RIGHT_ELBOW.value,0] = 1
    test_key_points[PifPafKeyPoints.RIGHT_ELBOW.value,1] = 2


    pifpaf_det = PifPafDetector()
    
    detected = pifpaf_det.check_if_desired_pose(test_key_points)
    
    assert True


if __name__ == "__main__":
    test_check_if_desired_pose()