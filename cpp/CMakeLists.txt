# CMakeLists.txt —— 编译 dcn_infer 与 detect(纯 C++ TensorRT 部署)。
#
# 使用本地 TRT 头文件 + pip 安装的 libnvinfer.so;插件在运行期 dlopen,不在此链接。
#
# 编译(WSL,conda activate dl,在 cpp/ 目录):
#   TRT_INC=$(realpath ../../trt_quantize/TensorRT-10.16/include)
#   NVINFER=$(python -c "import tensorrt_libs,glob,os; print(glob.glob(os.path.join(os.path.dirname(tensorrt_libs.__file__),'libnvinfer.so*'))[0])")
#   cmake -B build -DTRT_INCLUDE_DIR=$TRT_INC -DTRT_NVINFER=$NVINFER
#   cmake --build build -j
#
# 运行(在 python/ 目录,引擎位于此处;先 export 使可执行文件能找到 pip 的 libnvinfer):
#   export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))"):$LD_LIBRARY_PATH
#   ../cpp/build/dcn_infer backbone_fp16.engine ../src/build/libdcnv2.so

cmake_minimum_required(VERSION 3.18)            # 需 CMake 3.18 及以上
project(dcn_infer LANGUAGES CXX)                 # 仅启用 C++(本目录无 .cu,不需要 CUDA 编译器)

set(CMAKE_CXX_STANDARD 17)                       # C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)              # 强制 C++17,不降级
if(NOT CMAKE_BUILD_TYPE)                          # 未指定构建类型时
  set(CMAKE_BUILD_TYPE Release)                  # 默认 Release(延迟基准需启用优化)
endif()

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 根目录(含 include/ 与 lib/)")  # 允许 -DTENSORRT_ROOT=... 指定 TRT 根目录
find_path(TRT_INCLUDE_DIR NvInfer.h              # 查找 TRT 头文件(NvInfer.h)所在目录
  HINTS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include)
find_library(TRT_NVINFER NAMES nvinfer libnvinfer.so.10   # 查找 TRT 核心库
  HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu)
if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER)        # 头文件或库任一缺失
  message(FATAL_ERROR "找不到 TensorRT。请指定 -DTRT_INCLUDE_DIR=/abs/include -DTRT_NVINFER=/abs/libnvinfer.so.10")
endif()
message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")  # 打印实际使用的头文件路径

find_package(CUDAToolkit REQUIRED)               # 查找 CUDA Toolkit(需 CUDA::cudart 提供 cudaMalloc/Event 等运行时调用)
find_package(OpenCV QUIET)                        # 可选:安装 OpenCV 后 detect 才支持直接读图/画图(否则退回 bin 模式)

add_executable(dcn_infer dcn_infer.cpp)          # 可执行文件 dcn_infer
target_include_directories(dcn_infer PRIVATE ${TRT_INCLUDE_DIR})  # 添加 TRT 头搜索路径
# ${CMAKE_DL_LIBS} 即 -ldl(dlopen 所需)
target_link_libraries(dcn_infer PRIVATE ${TRT_NVINFER} CUDA::cudart ${CMAKE_DL_LIBS})  # 链接 TRT 核心库 + CUDA 运行时 + dl;不链接 libdcnv2.so,插件运行期 dlopen

# detect:纯 C++ 端到端检测(TrtEngine + RetinaNet 解码 + class-aware NMS,实现见 detect_main.cpp / dcn_detector.h / trt_engine.h)
add_executable(detect detect_main.cpp)
target_include_directories(detect PRIVATE ${TRT_INCLUDE_DIR})
target_link_libraries(detect PRIVATE ${TRT_NVINFER} CUDA::cudart ${CMAKE_DL_LIBS})
if(OpenCV_FOUND)                                  # 找到 OpenCV 则启用图片读写能力
  target_include_directories(detect PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(detect PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(detect PRIVATE HAVE_OPENCV)   # 源码以 #ifdef HAVE_OPENCV 启用图片分支
  message(STATUS "已找到 OpenCV:detect 支持直接读取图片(imread -> 预处理 -> 画框 -> det_out.jpg)")
else()
  message(STATUS "未找到 OpenCV:detect 仅支持 det_input.bin 对齐模式(安装 OpenCV 后重新 cmake 即可读图)")
endif()
