// dcn_kernel.h —— DCNv2 前向 kernel 的对外 C++ 入口声明。
// 仅声明 launcher;__global__ kernel 实现见 dcn_kernel.cu。
// 插件(dcn_plugin.cpp)与独立单元测试(test_dcn*.cu)均通过本头文件调用 kernel。
#pragma once
#include <cuda_runtime.h>  // cudaStream_t 等运行时类型
#include <cuda_fp16.h>     // __half(FP16)
#include <cstdint>         // int8_t(INT8 路径)

// FP32 朴素前向:x/offset/mask/weight/bias/y 均为 float。
// 参数:形状(N/Cin/Cout/H/W/Ho/Wo)+ 卷积超参(K/stride/pad/dil)+ CUDA 流。
void dcnv2_launch(const float* x, const float* offset, const float* mask,
                  const float* weight, const float* bias, float* y,
                  int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                  int K, int stride, int pad, int dil, cudaStream_t stream);

// FP16 朴素前向:I/O 为 __half 以节省显存与带宽;kernel 内部仍以 float 累加保证精度。
void dcnv2_launch_half(const __half* x, const __half* offset, const __half* mask,
                       const __half* weight, const __half* bias, __half* y,
                       int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                       int K, int stride, int pad, int dil, cudaStream_t stream);

// INT8 混合精度前向:x 与 weight 为 INT8(各带一个 per-tensor scale 用于反量化),
// offset/mask/bias 保持 FP32(几何量不量化),输出为 FP32。
void dcnv2_launch_int8(const int8_t* x, float x_scale,
                       const float* offset, const float* mask,
                       const int8_t* weight, float w_scale, const float* bias, float* y,
                       int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                       int K, int stride, int pad, int dil, cudaStream_t stream);

// 快速路径(im2col + GEMM,仅支持 N=1):先将每个采样值计算一次并写入
// cols[Cin*K*K, Ho*Wo],再由 cuBLAS 计算 weight[Cout,Cin*K*K] @ cols,
// 从而消除朴素实现中按输出通道重复采样的 Cout 倍冗余。
void deform_im2col_launch(const float* x, const float* offset, const float* mask, float* cols,
                          int Cin, int H, int W, int Ho, int Wo,
                          int K, int stride, int pad, int dil, cudaStream_t stream);  // 不含 N 参数:仅支持单 batch
// GEMM 之后,将 bias[oc] 广播加到 out[Cout,HW]。
void add_bias_launch(float* out, const float* bias, int Cout, int HW, cudaStream_t stream);
