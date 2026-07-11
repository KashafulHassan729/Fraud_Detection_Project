cmake_minimum_required(VERSION 3.18)
project(FraudDetectionSystem LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(SOURCES
    src/main.cpp
    src/preprocessing.cpp
    src/decision_module.cpp
    src/distance_calculation.cu
)

add_executable(fraud_detector ${SOURCES})

target_link_libraries(fraud_detector
    PRIVATE
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_include_directories(fraud_detector PRIVATE src)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set_target_properties(fraud_detector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output
)