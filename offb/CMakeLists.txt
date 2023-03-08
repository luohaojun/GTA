cmake_minimum_required(VERSION 3.0.2)
project(offb)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  roscpp
  std_msgs
  cv_bridge
  image_transport
  sensor_msgs
  message_generation
  genmsg
  visualization_msgs
  pcl_ros
  message_filters
)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED)

set(ncnn_DIR "/home/luo/gta/src/GTA/offb/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
if (${ncnn_FOUND})
  message("-- NCNN found.")
  message("-- NCNN_VULKAN flag is ${NCNN_VULKAN}")
  if (${NCNN_VULKAN})
    message("-- AUTO-ENABLING GPU_SUPPORT")
    set(GPU_SUPPORT OFF)
  endif()
  include_directories(${ncnn_INCLUDE})
endif()

add_message_files(
 FILES
 obj.msg
)


generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES offb
  CATKIN_DEPENDS geometry_msgs mavros_msgs roscpp std_msgs message_generation message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


add_executable(track
    src/track.cpp
    src/include/movement.cpp
    src/include/run_yolo.cpp)

target_link_libraries(track
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES})

add_executable(camera
    src/camera.cpp
    src/include/movement.cpp
    src/include/run_yolo.cpp)

target_link_libraries(camera
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES})

add_executable(obj_pos_world
   src/obj_pos_world.cpp
   src/include/movement.cpp
   src/include/run_yolo.cpp)

target_link_libraries(obj_pos_world
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES})

add_executable(gps
    src/gps.cpp
    src/include/movement.cpp
    src/include/run_yolo.cpp)

target_link_libraries(gps
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES})

add_executable(record
    src/record.cpp
    src/include/movement.cpp
    src/include/run_yolo.cpp)

target_link_libraries(record
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES})

add_executable(traj
    src/traj.cpp
    src/include/movement.cpp
    src/include/run_yolo.cpp)

target_link_libraries(traj
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES})

add_executable(yolo
   src/yolo.cpp
   src/include/movement.cpp
   src/include/run_yolo.cpp)

target_link_libraries(yolo
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES})

add_executable(camera_and_lidar
   src/camera_and_lidar.cpp
   src/include/movement.cpp
   src/include/run_yolo_only.cpp)

target_link_libraries(camera_and_lidar
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES})

add_executable(camera_and_lidar_2
   src/camera_and_lidar_2.cpp
   src/include/movement.cpp
   src/include/run_yolo_only.cpp)

target_link_libraries(camera_and_lidar_2
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES})

add_executable(syn
   src/syn.cpp
   src/include/movement.cpp
   src/include/run_yolo_only.cpp)
target_link_libraries(syn
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES})

add_executable(yolo_ncnn
   src/ncnn_yolo.cpp
   src/run_ncnn.hpp)

target_link_libraries(yolo_ncnn
    ncnn
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES})

add_executable(yolo_ncnn_2.0
   src/ncnn_yolo_2.0.cpp
   src/include/run_ncnn_2.0.cpp)
target_link_libraries(yolo_ncnn_2.0
    ncnn
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES})


