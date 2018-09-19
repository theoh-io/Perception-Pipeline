cmake_minimum_required(VERSION 3.10)
project(dynav-loomo)

set(CMAKE_CXX_STANDARD 11)

SET(GCC_COVERAGE_COMPILE_FLAGS "-I/usr/include/python3.6m -I/usr/include/python3.6m  -Wno-unused-result -Wsign-compare -g -fdebug-prefix-map=/build/python3.6-EKG1lX/python3.6-3.6.5=. -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector-strong -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC")
SET(GCC_COVERAGE_LINK_FLAGS    "-L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu -L/usr/lib -lpython3.6m -lpthread -ldl  -lutil -lm  -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

#include_directories(/home/changan/.pyenv/versions/py3/include)
#link_directories(/home/changan/.pyenv/versions/py3/lib)
include_directories(/usr/include/python3.6/)
link_directories(//usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/)

add_executable(dynav-loomo main.cpp PyUtils.cpp)

target_link_libraries(dynav-loomo libpython3.6.so)
