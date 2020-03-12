# Loomo Socket

This repository provides a socket protocol that supports real-time data communication between a Loomo robot and a remote computer, such as a laptop or a server machine. 

# Install Dependency
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
make -j8
make install
```

# Test Locally
```
cd cpp
mkdir build 
cd build 
cmake ..
make -j8
```

# Loomo Deployment
Connect to Loomo via adb
```
adb connect [port]
adb devices
```