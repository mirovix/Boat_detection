cmake_minimum_required(VERSION 2.8)
project(detecting_boats)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detecting_boats detecting_boats.cpp detecting_boats.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

file(COPY model DESTINATION ${CMAKE_BINARY_DIR})

file(COPY python_files DESTINATION ${CMAKE_BINARY_DIR})
