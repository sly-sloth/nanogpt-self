cmake_minimum_required(VERSION 3.14)
project(bytoken)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Define the Python module
pybind11_add_module(bytoken
    bytoken_binding.cpp
    bytoken.cpp
)
