cmake_minimum_required(VERSION 3.10)

# Tags 
SET(USE_CLIENT 1)
SET(USE_SERVER 1)

# Build
SET(PROJECT_NAME socket)
SET(CMAKE_CXX_STANDARD 11)
MESSAGE("@ Source path: " ${PROJECT_SOURCE_DIR})
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
find_package(OpenCV 3.4 REQUIRED PATHS ${PROJECT_SOURCE_DIR}/../dependency)

# General 
# include_directories(/usr/include /usr/local/include)

# Specific 
IF (USE_CLIENT)
	# Folder 
	include_directories(
		${PROJECT_SOURCE_DIR}
	)
	# File 
	add_executable( client
		${PROJECT_SOURCE_DIR}/SocketClient.cpp
		${PROJECT_SOURCE_DIR}/SampleClient.cpp
	)
	# Dependency 
	target_link_libraries( client 
		pthread 
		${OpenCV_LIBS}
	)
ENDIF()

IF(USE_SERVER)
	# Folder 
	include_directories(
		${PROJECT_SOURCE_DIR}
	)
	# File 
	add_executable(server
		${PROJECT_SOURCE_DIR}/SocketServer.cpp
		${PROJECT_SOURCE_DIR}/SampleServer.cpp
	)
	# Dependency 
	target_link_libraries( server 
		pthread 
		${OpenCV_LIBS}
	)
ENDIF()


