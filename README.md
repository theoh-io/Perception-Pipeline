# **DLAV Project**

## Intro
This repository shows the implementation of a robust perception pipeline supporting real-time on Loomo. Furthermore, this Repository can also serve as a benchmark to test the performance of different perception algorithms on Single Person Tracking Videos/Img Sequences.

For the pipeline our proposed methode combines **Yolov5** and **OpenPifPaf** for the Detection **Deepsort** for the Tracking and **Deep Visual Re-Identification with Confidence** for the ReID.

## User Guide
### Prerequisites
    - Python3
    - OpenCV (See Dependency at the bottom to build from source)
    - Pytorch
    - torchvision

### Benchmarking
Launch the `benchmark.sh` script to try the default perception module on a benchmark img sequence  (make sure to have added execution permission with `sudo chmod +x`).

To try inference on other data from [Tracking Benchmark](www.visual-tracking.net) just download and unzip in Benchmark folder, then change the name_file variable in `benchmark.sh`.

*/!\ Some ground truth doesn't use the same format (Blur Body...), just add the separator type ('\t') in inference.py l.309*

### Connection with Loomo
Make sure to be connected to the same wifi, get the ip adress of Loomo from settings. Then connect using `adb connect <ip Loomo>`  and check connection is working using  `adb devices`.

Run the AlgoApp on Loomo and press on button **Start Algo** to allow for socket connection, you should now see the camera on Loomo's screen.

### QuickStart with Loomo
Easiest way to start is to change the ip-adress (changing each time) of loomo in the bash script (`run_yolo.sh` / `run_yolo+tracker.sh` /...)  and launch the script containing basic options for the algorithm
`./run.sh`  (make sure to have added execution permission with `sudo chmod +x`)

---

## Repo structure

    .
    ├── Benchmark      # Img sequences + groundtruth bbox
    ├── python         # Python files containing perception module
    ├── Results        # Folder to store the videos resulting from inference
    ├── docs           # Basic docs for tandem race
    ├── cpp            # Cpp files on the robot side
    ├── scripts        # Basic bash scripts for configuration
    ├── others         # Other materials / Android app APK
    └── README.md

### Options
`-s <input video source>` by default loomo stream but can also be used by passing a recorded video (to compare trackers for example)

`-i <ip-address Loomo>` used by socket to communicate with Loomo.

`-c <checkpoints>` path to the weights of the tracker. if no path is provided then no tracker is used.

`-d <downscale parameter>` impact the resolution of the video stream 2 by default and mus be changed accordingly in `script/follow.cfg` then pushed to Loomo using `adb push follow.cfg /sdcard/`

---
#### Options for detector
`-yt <float in [0;1]>` yolo-threshold value for confidence in detection.

`-ym <name of the yolo model: yolov5n, yolov5s, yolov5m...>` Define the yolov5 pretrained model to use. If not locally stored will automatically download it online.

---
#### Options for tracker
`--dist-metric <'L2', 'cosine'>` metric to compute distance between embeddings in the tracker

`-tt <tracker distance threshold>` distance threshold meaning that embedding refers to the same target as the reference (/!\ L2 distance must be below threshold, while cosine similarity must be above).

`ref-emb <method to get reference embedding: multiple, simple, smart` multiple will keep fixed number of ref embedding and average across them while simple will only keep the last embedding. Finally smart will try to keep a limited number of high diversity embeddings.

`--nb_ref <number of ref embeddings>` Max number of embeddings to keep and average across when using multiple or smart

`--av-method <averaging method name>` Weights used during the averaging of the the reference embedding list, according more weights on recent detection (standard, linear, exponential). Use standard by default <=> all detections having the same weight.

`--intra-dist <L2 distance threshold for smart>` Above this threshold the embedding in the list of references will be kept to store high diversity representation


## Structure

```
socket-loomo/python
│
│─── run.sh ───> bash script to launch algorithm with basic options
│
│─── client.py ───> main script to launch the algorithm
│
│─── detector.py ───> Definition of the class for each detector (Yolov5)
│
│─── tracker.py ───> Definition of the class for each tracker (Re-ID)
│
│─── test.py ───> simple script to test the video stream with Loomo
│
│─── ReID_model.pth.tar ───> weights of the Re-ID tracker trained on Market1501 dataset with confidence penalty
│
│─── yolov5s.pt ───> Pretrained weights for Yolov5s used for real-time detection
```

## Dependency

The code and examples are tested on Linux. 

```
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
```

## Citations

## Other resources

### Introductory documents

Please find the following [documents](docs) for an introduction to the Loomo robot and a socket protocol.

* Getting_Started_with_Loomo.pdf
* Environment_Setup_Robots.pdf
* Loomo_Deployment_Instruction.pdf
* Tutorial_Loomo_Race.pdf

* [Loomo with Ros](https://github.com/cconejob/Autonomous_driving_pipeline)
* [Loomo Follower App](https://github.com/segway-robotics/loomo-algodev/blob/master/algo_app/src/main/jni/app_follow/AlgoFollow.cpp)

Other options are available: such as the choice of the detection algorithm and the tracker being used.

    python3 client.py --options

### Local test

Compile the server in the cpp folder
```
compile_socket.sh
```

Run the server in the cpp folder
```
run_server.sh
```

Run the client in the python folder
```
run_client.sh
```
