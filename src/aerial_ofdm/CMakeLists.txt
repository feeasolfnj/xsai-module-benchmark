cmake_minimum_required(VERSION 3.24)
project(aerial_ofdm_rx_benchmark LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG PATHS "/usr/local/nvidia/mathdx/26.03/")

add_executable(aerial-ofdm-rx
    aerial_ofdm_rx.cpp
    "${AERIAL_CHAN_SRC}/ofdm_src/ofdmDemod.cu"
    "${AERIAL_CHAN_SRC}/gpu3gppchanCommon.cpp")
target_include_directories(aerial-ofdm-rx PRIVATE
    "${AERIAL_CHAN_SRC}/ofdm_src" ${HDF5_INCLUDE_DIRS})
target_link_libraries(aerial-ofdm-rx PRIVATE
    CUDA::cudart CUDA::cuda_driver CUDA::curand mathdx::cufftdx ${HDF5_C_LIBRARIES})
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "121-real")
    target_compile_definitions(aerial-ofdm-rx PRIVATE AERIAL_BENCH_SM121)
endif()
if(AERIAL_BENCH_SM_PLACEMENT)
    target_compile_definitions(aerial-ofdm-rx PRIVATE AERIAL_BENCH_SM_PLACEMENT)
endif()
