# BreaKHis Classification with GradCAM
> Fine-tuning classification networks and using GradCAM to obtain object detection results on histopathological slides of breast cancer.


## Install

`pip install breakhis_gradcam`

## Later

* Examples of how to use this package
* Using some form of unsupervised bounding box detection (e.g. attention cropping) to pull bounding box predictions
    * This is probably more reasonable to do with a dataset that has these labels, as more training can be done to get better bounding box predictions
* Using a variant of the SimCLR framework to fine-tune the trained models based on similarity between detected regions of slides within the same subclass


## Contact

dt372@cornell.edu
