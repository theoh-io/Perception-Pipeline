# dynav-loomo
Run pytorch pretrained model on loomo

1. opencv 
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
2. 
```
cd socket
mkdir build 
cd build 
cmake ..
make -j8
```
3. Connect to Loomo via adb connect
```
adb connect [port]
adb devices
```