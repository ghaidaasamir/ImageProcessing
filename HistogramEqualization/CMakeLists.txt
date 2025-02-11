cmake_minimum_required(VERSION 3.10)

project(HistogramEqualization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(HistogramEqualization HistogramEqualization.cu)

target_link_libraries(HistogramEqualization ${OpenCV_LIBS})

target_include_directories(HistogramEqualization PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)