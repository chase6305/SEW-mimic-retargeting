cmake_minimum_required(VERSION 3.18)
project(sew_mimic_cpp LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_sew_mimic_cpp sew_mimic_cpp.cpp)
target_compile_features(_sew_mimic_cpp PRIVATE cxx_std_17)
target_compile_options(
  _sew_mimic_cpp
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>
)

install(TARGETS _sew_mimic_cpp LIBRARY DESTINATION sew_mimic)
