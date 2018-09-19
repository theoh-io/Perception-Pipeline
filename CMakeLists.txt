cmake_minimum_required(VERSION 3.5)
project(dynav-loomo)

set(CMAKE_CXX_STANDARD 11)

SET(GCC_COVERAGE_COMPILE_FLAGS "-fPIC")
SET(GCC_COVERAGE_LINK_FLAGS    "")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")


#include_directories(/home/changan/.pyenv/versions/py3/include)
#link_directories(/home/changan/.pyenv/versions/py3/lib)
include_directories(/usr/include/python3.5/)
link_directories(/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/)

add_executable(dynav-loomo main.cpp PyUtils.cpp)

target_link_libraries(dynav-loomo libpython3.5.so)
