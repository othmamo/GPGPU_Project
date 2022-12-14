cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

option(PROF "Reference build" NO)
include(FetchContent)


FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )


FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
# find_package(PNG REQUIRED)
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)

#set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


include_directories(${PROJECT_SOURCE_DIR}/includes)

add_subdirectory(src)

add_library(utilsimage)
target_compile_features(utilsimage PUBLIC cxx_std_14)
target_sources(utilsimage PRIVATE src/helpers_images.cpp)


##### CPU

add_library(detector_cpu)
target_compile_features(detector_cpu PUBLIC cxx_std_14)
target_link_libraries(detector_cpu PRIVATE spdlog::spdlog utilsimage blurring_cpu openclose_cpu threshold_cpu bbox_cpu)
target_sources(detector_cpu PRIVATE src/cpu/detect_obj_cpu.cpp src/struct_utils.cpp)

##### GPU

add_library(detector_gpu)
target_compile_features(detector_gpu PUBLIC cxx_std_14)
target_link_libraries(detector_gpu PRIVATE spdlog::spdlog utilsimage threshold_gpu bbox_gpu
    gray_gpu helpers_gpu blurr_gpu diff_gpu openclose_gpu)
target_sources(detector_gpu PRIVATE src/gpu/detect_obj_gpu.cu)
set_target_properties(detector_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_library(gpu_test_functions)
# target_compile_features(gpu_test_functions PUBLIC cxx_std_14)
# target_link_libraries(gpu_test_functions PRIVATE utilsimage)
# target_sources(gpu_test_functions PRIVATE src/test_gpu/gpu_functions.hpp src/test_gpu/gpu_functions.cu)
# set_target_properties(gpu_test_functions PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


##### MAIN
# <<<<<<< HEAD
# add_executable(main_gpu_test src/test_gpu/main_gpu_test.cpp)
# target_compile_features(main_gpu_test PUBLIC cxx_std_17)
# target_link_libraries(main_gpu_test PRIVATE gpu_test_functions)

# add_executable(main_cpu src/main_cpu.cpp)
# =======

add_executable(main_cpu src/cpu/main_cpu.cpp)
target_compile_features(main_cpu PUBLIC cxx_std_17)
target_include_directories(main_cpu PRIVATE "includes")
target_link_libraries(main_cpu CLI11::CLI11 spdlog::spdlog detector_cpu utilsimage)

add_executable(main_gpu src/gpu/main_gpu.cpp)
target_compile_features(main_gpu PUBLIC cxx_std_17)
target_include_directories(main_gpu PRIVATE "includes")
target_link_libraries(main_gpu CLI11::CLI11 spdlog::spdlog detector_gpu utilsimage)

##### BENCH
add_executable(bench_cpu src/bench/bench_cpu.cpp)
target_compile_features(bench_cpu PUBLIC cxx_std_17)
target_include_directories(bench_cpu PRIVATE "includes")
target_link_libraries(bench_cpu benchmark::benchmark detector_cpu utilsimage)

add_executable(bench_gpu src/bench/bench_gpu.cpp)
target_compile_features(bench_gpu PUBLIC cxx_std_17)
target_include_directories(bench_gpu PRIVATE "includes")
target_link_libraries(bench_gpu benchmark::benchmark detector_gpu utilsimage)
