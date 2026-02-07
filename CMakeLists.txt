cmake_minimum_required(VERSION 3.10)
project(rrtx_cpp_visualizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -march=native)
endif()

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

set(SOURCE_FILES
    main.cpp
    utils_cpp/RRTx.cpp
)

include_directories(utils_cpp/header)
pybind11_add_module(rrtx_cpp ${SOURCE_FILES})
message(STATUS "Configured to build rrtx_cpp python extension")