// test_dcn_half_gemm.cu —— 独立验证 FP16 快速路径(half im2col + cublasGemmEx tensor core + add_bias)。
//   读取 FP32 bins,转为 __half 运行 FP16 im2col-GEMM 路径,结果转回 float 与 oracle 比对 + cudaEvent 计时。
//   FP16 仅用于存储 I/O,GEMM 累加走 FP32(CUBLAS_COMPUTE_32F),故误差应远小于 INT8。
//
// 编译运行(在 python/ 目录,.bin 位于此处;需链接 cuBLAS):
//   nvcc -O2 -std=c++17 -arch=sm_89 ../src/test_dcn_half_gemm.cu ../src/dcn_kernel.cu -I../src -lcublas -o test_dcn_half_gemm
//   ./test_dcn_half_gemm

#include "dcn_kernel.h"   // deform_im2col_half_launch / add_bias_half_launch
#include <cublas_v2.h>    // cuBLAS(cublasGemmEx)
#include <cuda_fp16.h>    // __half / __float2half / __half2float
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){                 \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)  // CUDA 错误检查宏

// 从裸 .bin 读取 n 个 float。
static std::vector<float> load(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "无法打开 %s(请先运行 01_oracle_and_export.py)\n", path); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "%s 大小不符\n", path); exit(1); }
    fclose(f);
    return v;
}

// host 端 float -> __half 批量转换(cuda_fp16 的 __float2half 为 __host__ __device__)。
static std::vector<__half> to_half(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
    return h;
}

int main() {
    const int Cin=16, Cout=16, H=32, W=32, K=3, stride=1, pad=1, dil=1, Ho=32, Wo=32;  // 形状/超参(同 01;N=1)
    const int CKK = Cin*K*K, HW = Ho*Wo;               // CKK=im2col 行数(=weight 列数);HW=输出空间点数
    auto x   = load("dcn_x.bin",      (size_t)Cin*H*W);
    auto off = load("dcn_offset.bin", (size_t)2*K*K*Ho*Wo);
    auto msk = load("dcn_mask.bin",   (size_t)K*K*Ho*Wo);
    auto w   = load("dcn_weight.bin", (size_t)Cout*Cin*K*K);   // 视作 [Cout, CKK]
    auto b   = load("dcn_bias.bin",   (size_t)Cout);
    auto yref= load("dcn_y.bin",      (size_t)Cout*Ho*Wo);     // 视作 [Cout, HW] 的 oracle(FP32)

    auto xh=to_half(x), oh=to_half(off), mh=to_half(msk), wh=to_half(w), bh=to_half(b);  // 全部转 __half

    __half *dx,*doff,*dmsk,*dw,*db,*dy,*dcols;          // I/O 与 cols 均为 __half
    CK(cudaMalloc(&dx, xh.size()*2));   CK(cudaMalloc(&doff, oh.size()*2));  // *2 = sizeof(__half)
    CK(cudaMalloc(&dmsk, mh.size()*2)); CK(cudaMalloc(&dw, wh.size()*2));
    CK(cudaMalloc(&db, bh.size()*2));   CK(cudaMalloc(&dy, yref.size()*2));
    CK(cudaMalloc(&dcols, (size_t)CKK*HW*2));           // cols[CKK,HW](__half)
    CK(cudaMemcpy(dx, xh.data(), xh.size()*2, cudaMemcpyHostToDevice));      // H2D(__half)
    CK(cudaMemcpy(doff,oh.data(),oh.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmsk,mh.data(),mh.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, wh.data(), wh.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, bh.data(), bh.size()*2, cudaMemcpyHostToDevice));

    cublasHandle_t h; cublasCreate(&h);
    float one=1.f, zero=0.f;                            // COMPUTE_32F 的 alpha/beta 为 float
    // FP16 快速路径:half im2col -> cublasGemmEx(FP16 输入 + FP32 累加 + tensor core)-> 加 bias。
    auto run = [&]() {
        deform_im2col_half_launch(dx, doff, dmsk, dcols, Cin, H, W, Ho, Wo, K, stride, pad, dil, 0);  // 1) cols=m*bilinear(x_half)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, HW, Cout, CKK,                                      // 2) out=weight@cols
                     &one, dcols, CUDA_R_16F, HW,
                           dw,    CUDA_R_16F, CKK,
                     &zero, dy,   CUDA_R_16F, HW,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        add_bias_half_launch(dy, db, Cout, HW, 0);                                                    // 3) 广播加 bias
    };
    run();                                             // 正确性比对跑一次
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<__half> yh(yref.size());
    CK(cudaMemcpy(yh.data(), dy, yref.size()*2, cudaMemcpyDeviceToHost));   // D2H(__half)
    double maxerr=0, sumabs=0, sumref=0;               // 最大绝对误差 / 误差绝对值和 / 参考绝对值和
    for (size_t i=0;i<yh.size();++i){ double yi=__half2float(yh[i]);        // half -> float 再比对
        maxerr=fmax(maxerr,fabs(yi-yref[i])); sumabs+=fabs(yi-yref[i]); sumref+=fabs((double)yref[i]); }
    double rel = sumabs / (sumref + 1e-12);            // 平均相对误差 = Σ|err| / Σ|ref|
    printf("FP16 im2col+GEMM vs FP32 oracle: max|err|=%.4f  平均相对误差=%.2f%%\n", maxerr, rel*100);
    printf("%s\n", rel < 0.02 ? "[PASS] FP16 快速路径数值正确(FP16 存储 + FP32 累加)"
                              : "[FAIL] 相对误差偏大:检查 half im2col 或 GemmEx 的 compute type");

    // cudaEvent 计时(Cin=Cout=16 微规模,仅示意;消除 Cout 倍冗余的真实加速见 backbone 基准 dcn_infer / 03)。
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i=0;i<20;i++) run();                       // 预热
    CK(cudaDeviceSynchronize());
    cudaEventRecord(t0);
    for (int i=0;i<100;i++) run();                      // 计时 100 次取平均
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
    float ms=0; cudaEventElapsedTime(&ms, t0, t1);
    printf("FP16 im2col+GEMM 平均 %.4f ms/次(微规模,真实加速见 backbone 基准)\n", ms/100);

    cublasDestroy(h);
    cudaFree(dx);cudaFree(doff);cudaFree(dmsk);cudaFree(dw);cudaFree(db);cudaFree(dy);cudaFree(dcols);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return 0;
}
