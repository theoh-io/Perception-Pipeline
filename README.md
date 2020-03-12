# Loomo Socket

This repository provides a socket protocol that supports real-time data communication between a Loomo robot and a remote computer, such as a laptop or a server machine. The code and examples are tested on Linux.

### Directory hierarchy

    .
    ├── cpp                   # Cpp files on the robot side
    ├── python                # Python files on the cloud side
    ├── dependency            # Compile dependencies
    └── README.md

### Install Dependency

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

### Local Test

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

### Loomo Deployment
Connect to Loomo via adb
```
adb connect [port]
adb devices
```