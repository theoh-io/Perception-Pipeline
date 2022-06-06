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
currently: final_detector => perception.py (DetectorG16) at the root of perceptionloomo
localization of that top level class => src/perceptionloomo
name of the class: Detector G16 need to change
Forward method

## Downloading SOTA weights for REID
'resnet50_AGWmarket':
'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth',
'resnet50_AGWmarket':
https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50.pth',
'resnet50_SBSmsmt17':
'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50.pth',
need to rename them correctly

## To Do
- [x] same IO for reid_tracker
- [x] Create top level file + config file: running pifpaf frequency
- [x] Create script to download SOTA ReID weights and rename them appropriately
- [x] Add ds_custom_reid + compatibility with other ReID weights
- [x] config file
- [] make the check clean_tracking script working at Theos Laptop (theo)
- [] check client.py and add it in python/scripts (theo)
- [] [benchmarking performance IoU]
- [] add comment (documenting code)
- [] Use logging instead of verbose



