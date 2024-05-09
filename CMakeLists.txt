cmake_minimum_required(VERSION 3.10)
project(semantic_dsp_map)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++17 -g -O3 -ftree-vectorize -march=native)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  mask_kpts_msgs
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

find_package(VTK REQUIRED)
find_package(yaml-cpp REQUIRED)

add_definitions(${PCL_DEFINITIONS})

catkin_package()

include_directories(${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${YAML_CPP_INCLUDE_DIR})


add_executable(mapping_with_external_data src/mapping_with_external_data.cpp)
target_link_libraries(mapping_with_external_data ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES})


