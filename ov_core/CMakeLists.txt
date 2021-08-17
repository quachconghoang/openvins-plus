cmake_minimum_required(VERSION 2.8.8)

# Project name
project(ov_core)

# Find catkin (the ROS build system)
find_package(catkin QUIET COMPONENTS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs cv_bridge)

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 3 REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)

# Display message to user
message(WARNING "OPENCV VERSION: " ${OpenCV_VERSION})
message(WARNING "BOOST VERSION: " ${Boost_VERSION})

# If we will compile with aruco support
option(ENABLE_ARUCO_TAGS "Enable or disable aruco tag (disable if no contrib modules)" ON)
if (NOT ENABLE_ARUCO_TAGS)
    add_definitions(-DENABLE_ARUCO_TAGS=0)
    message(WARNING "DISABLING ARUCOTAG TRACKING!")
else()
    add_definitions(-DENABLE_ARUCO_TAGS=1)
endif()

# Describe catkin project
option(ENABLE_CATKIN_ROS "Enable or disable building with ROS (if it is found)" ON)
if (catkin_FOUND AND ENABLE_CATKIN_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs cv_bridge
            INCLUDE_DIRS src
            LIBRARIES ov_core_lib
    )
else()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
endif()


# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
)

#TORCH HERE-WE-GO
set(CMAKE_PREFIX_PATH "~/local")
find_package(Torch REQUIRED)
if(TORCH_FOUND)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    find_package( CUDA REQUIRED )
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75;-rdc=true;-use_fast_math")
    include_directories(
            ${CUDA_INCLUDE_DIRS}
            ${TORCH_INCLUDE_DIRS})
    list(APPEND thirdparty_libraries
            ${TORCH_LIBRARIES})
endif()

##################################################
# Make the core library
##################################################
add_library(ov_core_lib SHARED
        src/dummy.cpp
        src/init/InertialInitializer.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackAruco.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackKLT.cpp
        src/track/TrackSIM.cpp
        src/track/TrackTorch.cpp
        src/track/Superpoint.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureInitializer.cpp)
target_link_libraries(ov_core_lib ${thirdparty_libraries})
set_property(TARGET ov_core_lib PROPERTY CXX_STANDARD 14)
target_include_directories(ov_core_lib PUBLIC src)

##################################################
# Make binary files!
##################################################
#if (catkin_FOUND AND ENABLE_CATKIN_ROS)
#    add_executable(test_tracking src/test_tracking.cpp)
#    target_link_libraries(test_tracking ov_core_lib ${thirdparty_libraries})
#endif()

add_executable(test_tracking_viode src/test_tracking_viode.cpp)
target_link_libraries(test_tracking_viode ov_core_lib ${thirdparty_libraries})
#add_executable(test_webcam src/test_webcam.cpp)
#target_link_libraries(test_webcam ov_core_lib ${thirdparty_libraries})


