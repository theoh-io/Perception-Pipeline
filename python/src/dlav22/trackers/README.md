# Trackers Format

## List of available Trackers
reid_tracker (ReidTracker()): Theo's implementation averaging reference embeddings and comparing the distance to the ref from each detection
fused_ds_reid (FusedDsReid()) Combining Deepsort with another ReID model (currently custom tracker)
ds_custom_reid (DsCustomReid()): using custom ReID model inside Deepsort as Feature Extractor (option to load different ReID models)
[siamese (Siamese()): need to copy the files (create a lib) / submodules]

## Format of the trackers

Input: Cropped detection images, bbox_list, original_img
Output: bbox (x_center, y_center, w, h)
Class functions: track

## name of the Top Level Function
(currently final detector)
localization of that top level class
name of the class: Detector G16 need to change
Forward method