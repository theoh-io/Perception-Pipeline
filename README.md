# Loomo Socket

This repository contains a socket protocol that supports real-time data communication between a Loomo robot and a remote computer, such as a laptop or a server machine.

This sockek protocal, together with the associated documents, provides a basic setup for the [tandem race](https://actu.epfl.ch/news/robots-programmed-to-follow-you-4).

### Repo structure

    .
    ├── docs                  # Basic docs for tandem race
    ├── cpp                   # Cpp files on the robot side
    ├── python                # Python files on the cloud side
    ├── scripts               # Basic bash scripts for configuration
    ├── others                # Other materials for a toy TP exercise
    └── README.md

### Introductory documents

Please find the following [documents](docs) for an introduction to the Loomo robot and a socket protocol.

* Getting_Started_with_Loomo.pdf
* Environment_Setup_Robots.pdf
* Loomo_Deployment_Instruction.pdf

### Install dependency

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

### On-robot deployment

Connect to Loomo via adb
```
adb connect <port>
adb devices
```

### Other resources

* [Loomo with Ros](https://github.com/cconejob/Autonomous_driving_pipeline)
* [Loomo Follower App](https://github.com/segway-robotics/loomo-algodev/blob/master/algo_app/src/main/jni/app_follow/AlgoFollow.cpp)