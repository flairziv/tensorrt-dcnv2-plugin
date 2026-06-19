# CMakeLists.txt —— 将 DCNv2 kernel 与插件编译为 libdcnv2.so(TensorRT 10 插件)。
#
# 产物:libdcnv2.so = dcn_kernel.cu(nvcc)+ dcn_plugin.cpp(g++,IPluginV3)。
#
# 编译命令(WSL,conda activate dl,在 src/ 目录):
#   TRT_INC=$(realpath ../../trt_quantize/TensorRT-10.16/include)
#   NVINFER=$(python -c "import tensorrt_libs,glob,os; print(glob.glob(os.path.join(os.path.dirname(tensorrt_libs.__file__),'libnvinfer.so*'))[0])")
#   cmake -B build -DTRT_INCLUDE_DIR=$TRT_INC -DTRT_NVINFER=$NVINFER
#   cmake --build build -j
# 验证: cd ../python && python 02_build_and_validate.py

cmake_minimum_required(VERSION 3.18)           # 需 CMake 3.18 及以上(对 CUDA 架构与 CUDAToolkit 导入 target 支持更完善)
project(dcnv2_plugin LANGUAGES CXX CUDA)        # 同时启用 C++ 与 CUDA(.cpp 走 g++,.cu 走 nvcc)

set(CMAKE_CXX_STANDARD 17)                      # host 端 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)             # 强制 C++17,不静默降级
set(CMAKE_CUDA_STANDARD 17)                     # CUDA 端 host 代码同样按 C++17
if(NOT CMAKE_BUILD_TYPE)                         # 未指定构建类型时
  set(CMAKE_BUILD_TYPE Release)                 # 默认 Release(启用优化,延迟基准方有意义)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)                 # 未指定目标 GPU 计算架构(SM 版本)时
  set(CMAKE_CUDA_ARCHITECTURES 89)              # 默认 sm_89(Ada,RTX 4060 Laptop);更换显卡需修改此处
endif()

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 根目录(含 include/ 与 lib/)")  # 允许 -DTENSORRT_ROOT=... 指定 TRT 安装根
find_path(TRT_INCLUDE_DIR NvInfer.h             # 搜索包含 NvInfer.h 的目录,得到 TRT 头文件路径
  HINTS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include)  # 优先级:用户指定 > 系统目录
find_library(TRT_NVINFER NAMES nvinfer libnvinfer.so.10   # 搜索 TRT 核心库
  HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu)
if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER)       # 头文件或库任一缺失
  message(FATAL_ERROR "找不到 TensorRT。请手动指定 -DTRT_INCLUDE_DIR=/abs/include -DTRT_NVINFER=/abs/libnvinfer.so.10")
endif()
message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")  # 打印实际使用的头文件路径
message(STATUS "TensorRT nvinfer: ${TRT_NVINFER}")      # 打印实际使用的库路径

find_package(CUDAToolkit REQUIRED)              # 查找 CUDA Toolkit(提供 CUDA::cudart / CUDA::cublas)

add_library(dcnv2 SHARED dcn_kernel.cu dcn_plugin.cpp)        # 编译动态库 libdcnv2.so:CUDA kernel + IPluginV3 插件
target_include_directories(dcnv2 PRIVATE ${TRT_INCLUDE_DIR})  # 为该 target 添加 TRT 头搜索路径
target_link_libraries(dcnv2 PRIVATE ${TRT_NVINFER} CUDA::cudart CUDA::cublas)  # 链接:TRT 核心库 + CUDA 运行时 + cuBLAS(FP32 快速路径 GEMM)
