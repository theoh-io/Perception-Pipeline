from time import sleep
import time
import cv2
import numpy as np
import time

from pathlib import Path
import yaml
import os 
if str(os.getcwd())[-7:] == "scripts":
    os.chdir("..")

print(f"Current WD: {os.getcwd()}")


from dlav22.utils.utils import FrameGrab

from dlav22.perception import perception
from dlav22.utils.utils import Utils

from dlav22.deep_sort.utils.parser import get_config

if __name__ == "__main__":
    detector = perception.DetectorG16()
    # start video stream
    video_path=detector.cfg.PERCEPTION.BENCHMARK_FILE
    print(f"Using the video: {video_path} as input")
    grab = FrameGrab(mode="video", video=video_path)

    bboxes_to_save = []
    elapsed_time_list=[]
    while(True):
        img = grab.read_cap()
        if img is None:
            print("Stop reading.")
            break
        tic = time.perf_counter()
        #Call to the algorithm
        bbox = detector.forward(img)
        toc = time.perf_counter()
        print(f"Elapsed time for whole fordward pass: {(toc-tic)*1e3:.1f}ms")
        #record bbox and elapsed time
        bboxes_to_save.append(bbox)
        elapsed_time_list.append((toc-tic)*1e3)
        #Visualization
        if bbox is not None:
            Utils.visualization(img, bbox, (255, 0, 0), 2)
        else:
            pass
        print("bbox:", bbox)
        #To get result before the end quit the program with q instead of Ctrl+C
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break

    Utils.save_results(detector, bboxes_to_save)
    # if detector.cfg.RECORDING.SAVE_RESULTS:
    #     bboxes_to_save = [b if b is not None else np.zeros(4) for b in bboxes_to_save]
    #     bboxes_to_save = np.array(bboxes_to_save, dtype=np.int16)

    #     folder_str = detector.cfg.RECORDING.FOLDER_FOR_PREDICTION
    #     # save_str = f"{folder_str}/{detector.cfg.PERCEPTION.BENCHMARK_FILE.replace('.','').replace('/','').replace('_','')}_tracker_{detector.cfg.TRACKER.TRACKER_CLASS[-11:]}.txt"
    #     save_str = f"{folder_str}/ID_{detector.cfg.RECORDING.EXPID:04d}_prediction"
    #     path = Path(f"{save_str}.txt")
    #     if path.is_file():
    #         print("File already exists. Did not store it.")
    #     else:
    #         print(f"Saving predicted bboxes to {save_str}.txt.")
    #         np.savetxt(f"{save_str}.txt", bboxes_to_save, fmt='%.i',delimiter=' , ')

    #         # FIXME Specify all parameters that are varied for a specif configuration
    #         config_dict = {"EXP_ID": detector.cfg.RECORDING.EXPID, "DS_MAX_DIST": detector.cfg.DEEPSORT.MAX_DIST}

    #         file = open(f"{save_str}.yaml", "w")
    #         yaml.dump(config_dict,file)
    #         file.close()
    #         detector.store_elapsed_time()

    average_forward_time=np.mean(elapsed_time_list)
    print(f"Average time for a forward pass is {average_forward_time:.1f}ms")




    cv2.destroyAllWindows()
    del grab
