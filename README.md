# dynav-loomo
Run pytorch pretrained model on loomo

1. opencv 
```
wget https://github.com/opencv/opencv/archive/3.4.5.zip
unzip 3.4.5.zip
mkdir opencv 
cd opencv-3.4.5
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../../opencv ..
make -j4
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