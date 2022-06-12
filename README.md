# **Perception-Pipeline SOT**

<p float="left">
  <img src="./others/GIF1.gif" width="275" />
  <img src="./others/GIF2.gif" width="275" /> 
</p>


---
## Intro
This repository shows the implementation of a robust and modular Perception-Pipeline. This Perception module was developed to be implemented for real-time on Loomo Segway Robot. Furthermore, this Repository can also serve as a benchmark to test the performance of different perception algorithms on Single Person Tracking Videos/Img Sequences.

This pipeline propose a modular implementation combining **Yolov5** and **OpenPifPaf** for the Detection module. **Deepsort**, **SiamRPN++** and **Stark** for the Tracking module and Finally **Deep Visual Re-Identification with Confidence** for the ReID.



## Prerequisites
 - Clone the repository recursively:

        git clone --recurse-submodules

    If you already cloned and forgot to use --recurse-submodules you can run `git submodule update --init`
- Look at the Install Procedure Described at the bottom
- Requirements:
    - Python3
    - OpenCV
    - Pytorch
    - torchvision
    - Openpifpaf

## Repo structure
    ├── Benchmark      # Videos/Img sequences + groundtruth bbox
    ├── libs     # Folder for the installation of external libraries (mmtracking, deep-person-reid)
    ├── python         # Python files containing perception module
    ├── Results        # Folder to store the videos and bbox resulting from inference
    ├── others         # Other materials / Android app APK
    └── README.md

Check the other ReadMe file (add link) to get more info about the perception module and configurations

## Downloading pretrained models

[Stark](https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/stark): Learning Spatio-Temporal Transformer for Visual Tracking

    cd src/perceptionloomo
    mkdir checkpoints
    wget https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth

[SiameseRPN++](https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/siamese_rpn):  Evolution of Siamese Visual Tracking With Very Deep Networks

    wget https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth

---
## Running the Pipeline on Video/Img sequence

### Downloading Benchmark Data
To download data to the Benchmark folder use the command:  `wget <link_data>`  in the Benchmark folder.
- [**LaSOT**](http://vision.cs.stonybrook.edu/~lasot/): Large scale dataset for Single Image Tracking (SOT) as our detection algorithm is only trained on humans we are only interested on the person category available for download [here](https://drive.google.com/drive/folders/1v09JELSXM_v7u3dF7akuqqkVG8T1EK2_?usp=sharing) 
- [**OTB100**](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html): Img sequences from various different sources with provided categories for tracking challenges (Illumination variations, Body deformation...). Only some sequences are relevant for out perception module (Human, Skater, BlurBody, Basketball...)
- **Loomo Dataset** provides 8 + 16 Videos with provided ground truth from real-life experiments recordings. The Dataset is available for download at this [link](https://drive.google.com/drive/folders/1r9-GRIsfojvwlljnHovZ5SvlmbUzsZtZ?usp=sharing)

### Inference on Benchmark Data
Launch the `benchmark.py` script (python/scripts) to try the default perception module configuration

(To get more details on how to change the configurations and input file check the ReadMe.md inside python directory)

---
## Running the Pipeline on Loomo
### Connection with Loomo
Make sure to be connected to the same WiFi, get the ip adress of Loomo from settings. Then connect using `adb connect <ip Loomo>`  and check connection is working using  `adb devices`.

Run the AlgoApp on Loomo and press on button **Start Algo** to allow for socket connection, you should now see the camera on Loomo's screen.

Before trying to launch the app on loomo make sure to have the same downscale parameter on Loomo and in the config file: `cfg_perception.yaml`. To see the config on loomo use the command: `adb shell cat /sdcard/follow.cfg`

### QuickStart with Loomo
Easiest way to start is to change the ip-adress of loomo in the config file (`python/src/perceptionloomo/configs/cfg_perception.yaml`)  and launch the script to run the algorithm on loomo
`python3 python/scripts/loomo.py`


---
## Install

### Virtual environment Setup
First set up a python >= 3.7 virtual environment:
```
$ cd <desired-dir>
$ python3 -m venv <desired venv name>
$ activate desired venv name
```
Make sure to install python 3 in case you do not have it.

### Modules Install

In the root directory: after creating a virtual environment run the following commands to install pip packages
 `pip install -r requirements.txt`

In the root directory do to the libs directory and install the following repositories. (In case you didn't clone with submodule run: `git submodule update --init `to download submodule for deep-person-reid)

    cd libs
    cd deep-person-reid/
    pip install -e . (real install procedure)

    cd ../
    git clone git@github.com:open-mmlab/mmtracking.git
    cd mmtracking 

Follow the install instructions from the [official documentation](https://mmtracking.readthedocs.io/en/latest/install.html)

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
    pip install mmdet
    pip install -r requirements/build.txt
    pip install -v -e .


In the `Perception-Pipeline/python/` directory run the following command to install our custom perceptionloomo package `pip install -e .`

### OpenCV install from source

In case you have problems with the installation of opencv using pip you can try to build it directly from source using the following procedure.

    mkdir depencency 
    cd dependency
    wget https://github.com/opencv/opencv/archive/3.4.5.zip
    unzip 3.4.5.zip
    mkdir opencv 
    cd opencv-3.4.5
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../../opencv -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_V4L=ON -DWITH_OPENGL=ON -DWITH_CUBLAS=ON -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
    make -j4
    make install

---
## Other resources
### Introductory documents

Please find the following documents for an introduction to the Loomo robot and a socket protocol.

* [Getting_Started_with_Loomo.pdf](./others/Getting_Started_with_Loomo.pdf)
* [Environment_Setup_Robots.pdf](./others/Environment_Setup_Robots.pdf)
* [Loomo_Deployment_Instruction.pdf](./others/Loomo_Deployment_Instruction.pdf)
* [Tutorial_Loomo_Race.pdf](./others/Tutorial_Loomo_Race.pdf)

* [Loomo with Ros](https://github.com/cconejob/Autonomous_driving_pipeline)
* [Loomo Follower App](https://github.com/segway-robotics/loomo-algodev/blob/master/algo_app/src/main/jni/app_follow/AlgoFollow.cpp)




