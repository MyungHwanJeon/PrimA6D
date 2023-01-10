cmake_minimum_required(VERSION 2.8.3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(pose_optimization)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  realsense2_camera
  nav_msgs
  rosbag
  visualization_msgs
  franka_rpm_msgs


)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(Boost REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp std_msgs franka_rpm_msgs
    DEPENDS PCL
)



include_directories(
  include
  bop_renderer/src
  bop_renderer/3rd
  bop_renderer/3rd/glm
  ${PROJECT_SOURCE_DIR}/bop_renderer/osmesa-install/install/osmesa/include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}		
)

# OSMesa.
set(OSMESA_PREFIX "/opt/osmesa")
set(OSMESA_LIBRARIES ${OSMESA_PREFIX}/lib/libOSMesa32.so)
include_directories(${OSMESA_PREFIX}/include)

# llvm
set(LLVM_DIR "/opt/llvm")


# Header files.
set(HEADERFILES
        bop_renderer/src/Renderer.h
        bop_renderer/src/PythonWrapper.h
        bop_renderer/src/AbstractLoader.h
        bop_renderer/src/PlyLoader.h
        bop_renderer/src/GPUBuffer.h
        bop_renderer/src/Pose.h
        bop_renderer/src/Geometry.h
        bop_renderer/src/glutils/FrameBufferObject.h
        bop_renderer/src/glutils/GLSLProgram.h
        bop_renderer/src/glutils/gl_core_3_3.h
        bop_renderer/src/context/Context.h
        bop_renderer/src/Texture.h
        bop_renderer/src/Shaders.h
        )

# Source files.
set(SOURCEFILES
        bop_renderer/src/Renderer.cpp
        bop_renderer/src/AbstractLoader.cpp
        bop_renderer/src/PlyLoader.cpp
        bop_renderer/src/GPUBuffer.cpp
        bop_renderer/src/Pose.cpp
        bop_renderer/src/glutils/FrameBufferObject.cpp
        bop_renderer/src/glutils/GLSLProgram.cpp
        bop_renderer/src/glutils/gl_core_3_3.cpp
        bop_renderer/src/context/OSMContext.cpp
        bop_renderer/3rd/rply/rply.c
        bop_renderer/3rd/lodepng/lodepng.cpp
        )

add_library(
	bop_renderer
	SHARED
	${SOURCEFILES}
	)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

add_executable(graph_optimization src/graph_optimization.cpp src/object_renderer.cpp src/usrdef.cpp )
add_dependencies(graph_optimization ${catkin_EXPORTED_TARGETS} franka_rpm_msgs_gencpp)
target_link_libraries(graph_optimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} gtsam Eigen3::Eigen bop_renderer ${OSMESA_LIBRARIES})

add_executable(graph_optimization_with_prior src/graph_optimization_with_prior.cpp src/object_renderer.cpp src/usrdef.cpp )
add_dependencies(graph_optimization_with_prior ${catkin_EXPORTED_TARGETS} franka_rpm_msgs_gencpp)
target_link_libraries(graph_optimization_with_prior ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} gtsam Eigen3::Eigen bop_renderer ${OSMESA_LIBRARIES})

add_executable(visualization src/visualization.cpp src/usrdef.cpp)
add_dependencies(visualization ${catkin_EXPORTED_TARGETS} franka_rpm_msgs_gencpp)
target_link_libraries(visualization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} gtsam Eigen3::Eigen)

add_executable(hand_eye_calibration src/hand_eye_calibration_with_depth.cpp src/usrdef.cpp)
add_dependencies(hand_eye_calibration ${catkin_EXPORTED_TARGETS} franka_rpm_msgs_gencpp)
target_link_libraries(hand_eye_calibration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} gtsam Eigen3::Eigen)
