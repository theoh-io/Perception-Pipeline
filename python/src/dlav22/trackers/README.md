# Trackers Format

## List of available Trackers
reid_tracker (ReidTracker()): Theo's implementation averaging reference embeddings and comparing the distance to the ref from each detection
fused_ds_reid (FusedDsReid()) Combining Deepsort with another ReID model (currently custom tracker)
ds_custom_reid (DsCustomReid()): using custom ReID model inside Deepsort as Feature Extractor (option to load different ReID models)
[siamese (Siamese()): need to copy the files (create a lib) / submodules]

## Format of the trackers

Input: Cropped detection images, bbox_list, original_img
Output: bbox (x_center, y_center, w, h), for ReID idx optionally as output
Class functions: track

## name of the Top Level Function
currently: final_detector => perception.py (DetectorG16) at the root of dlav22
localization of that top level class => src/dlav22
name of the class: Detector G16 need to change
Forward method

## To Do
same IO for reid_tracker ✅
Create top level file + config file: running pifpaf frequency
Add ds_custom_reid + compatibility with other ReID weights
config file
make the check clean_tracking script working
check client.py and add it in python/scripts
[benchmarking performance IoU]
add comment (documenting code)
