// test_dcn_int8_gemm.cu —— 独立验证 INT8 快速路径(整数域 im2col + 拓宽 weight + cuBLAS SGEMM + scale/bias)。
//   读取 FP32 bins,将 x/weight per-tensor 对称量化为 int8,运行 INT8 im2col-GEMM 路径并与 oracle 比对 + cudaEvent 计时。
//   与朴素 INT8(test_dcn_int8.cu)采用同一量化,故数值应一致;此处验证 im2col 重排未引入错误。
//
// 编译运行(在 python/ 目录,.bin 位于此处;需链接 cuBLAS):
//   nvcc -O2 -std=c++17 -arch=sm_89 ../src/test_dcn_int8_gemm.cu ../src/dcn_kernel.cu -I../src -lcublas -o test_dcn_int8_gemm
//   ./test_dcn_int8_gemm

#include "dcn_kernel.h"   // deform_im2col_int8_launch / dequant_weight_launch / add_bias_scale_launch
#include <cublas_v2.h>    // cuBLAS(cublasSgemm)
#include <cstdio>
#include <cstdlib>
#include <cmath>          // lroundf / fabsf / fmaxf
#include <cstdint>        // int8_t
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

// per-tensor 对称量化:scale = max|.|/127,四舍五入并截断到 [-127,127]。返回 scale(反量化时乘回)。
static float quantize(const std::vector<float>& src, std::vector<int8_t>& dst) {
    float mx = 0.f;
    for (float v : src) mx = fmaxf(mx, fabsf(v));
    float scale = (mx > 0.f) ? mx / 127.f : 1.f;       // 全 0 时兜底为 1 避免除零
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        int q = (int)lroundf(src[i] / scale);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);     // 对称量化不使用
        dst[i] = (int8_t)q;
    }
    return scale;
}

int main() {
    const int Cin=16, Cout=16, H=32, W=32, K=3, stride=1, pad=1, dil=1, Ho=32, Wo=32;  // 形状/超参(同 01;N=1)
    const int CKK = Cin*K*K, HW = Ho*Wo;               // CKK=im2col 行数(=weight 列数);HW=输出空间点数
    auto x   = load("dcn_x.bin",      (size_t)Cin*H*W);
    auto off = load("dcn_offset.bin", (size_t)2*K*K*Ho*Wo);
    auto msk = load("dcn_mask.bin",   (size_t)K*K*Ho*Wo);
    auto w   = load("dcn_weight.bin", (size_t)Cout*Cin*K*K);   // 视作 [Cout, CKK]
    auto b   = load("dcn_bias.bin",   (size_t)Cout);
    auto yref= load("dcn_y.bin",      (size_t)Cout*Ho*Wo);     // 视作 [Cout, HW] 的 oracle

    std::vector<int8_t> xq, wq;
    float xs = quantize(x, xq), ws = quantize(w, wq);  // x/weight 各自 per-tensor 对称量化,得到 scale
    printf("量化:x_scale=%.5f  w_scale=%.5f(per-tensor 对称 INT8)\n", xs, ws);

    int8_t *dx,*dw; float *doff,*dmsk,*db,*dy,*dcols,*dwf;  // x/weight 为 int8;cols/拓宽 weight/输出为 float
    CK(cudaMalloc(&dx, xq.size())); CK(cudaMalloc(&dw, wq.size()));         // int8 每元素 1 字节
    CK(cudaMalloc(&doff,off.size()*4)); CK(cudaMalloc(&dmsk,msk.size()*4));
    CK(cudaMalloc(&db, b.size()*4));    CK(cudaMalloc(&dy, yref.size()*4));
    CK(cudaMalloc(&dcols,(size_t)CKK*HW*4));            // cols[CKK,HW] = m*bilinear(x_int8)(整数幅值域)
    CK(cudaMalloc(&dwf,  (size_t)Cout*CKK*4));          // 拓宽(int8->float)后的 weight[Cout,CKK]
    CK(cudaMemcpy(dx, xq.data(), xq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(doff,off.data(),off.size()*4,cudaMemcpyHostToDevice));    // H2D FP32 几何量
    CK(cudaMemcpy(dmsk,msk.data(),msk.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(),  b.size()*4,  cudaMemcpyHostToDevice));

    cublasHandle_t h; cublasCreate(&h);
    float one=1.f, zero=0.f;                            // GEMM 系数 alpha=1, beta=0
    // INT8 快速路径四步:整数域 im2col -> 拓宽 weight -> SGEMM -> 施加 x_scale*w_scale 并加 bias。
    auto run = [&]() {
        deform_im2col_int8_launch(dx, doff, dmsk, dcols, Cin, H, W, Ho, Wo, K, stride, pad, dil, 0);  // 1) cols=m*bilinear(x_int8)
        dequant_weight_launch(dw, dwf, Cout*CKK, 0);                                                  // 2) weight int8->float(不乘 scale)
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, HW, Cout, CKK, &one, dcols, HW, dwf, CKK, &zero, dy, HW);
        add_bias_scale_launch(dy, db, xs*ws, Cout, HW, 0);                                            // 4) out=out*(xs*ws)+bias
    };
    run();                                             // 正确性比对跑一次
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<float> y(yref.size());
    CK(cudaMemcpy(y.data(), dy, yref.size()*4, cudaMemcpyDeviceToHost));    // D2H
    double maxerr=0, sumabs=0, sumref=0;               // 最大绝对误差 / 误差绝对值和 / 参考绝对值和
    for (size_t i=0;i<y.size();++i){ maxerr=fmax(maxerr,fabs((double)y[i]-yref[i]));
        sumabs+=fabs((double)y[i]-yref[i]); sumref+=fabs((double)yref[i]); }
    double rel = sumabs / (sumref + 1e-12);            // 平均相对误差 =
    printf("INT8 im2col+GEMM vs FP32 oracle: max|err|=%.4f  平均相对误差=%.2f%%\n", maxerr, rel*100);
    printf("%s\n", rel < 0.1 ? "[PASS] INT8 快速路径数值合理(应与朴素 INT8 约 1.27% 一致)"
                             : "[FAIL] 相对误差偏大:检查整数域 im2col / scale 施加位置");

    // cudaEvent 计时(Cin=Cout=16 微规模,仅示意;消除 Cout 倍冗余的真实加速见 backbone 基准 dcn_infer / 03)。
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i=0;i<20;i++) run();                       // 预热
    CK(cudaDeviceSynchronize());
    cudaEventRecord(t0);
    for (int i=0;i<100;i++) run();                      // 计时 100 次取平均
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
    float ms=0; cudaEventElapsedTime(&ms, t0, t1);
    printf("INT8 im2col+GEMM 平均 %.4f ms/次(微规模,真实加速见 backbone 基准)\n", ms/100);

    cublasDestroy(h);
    cudaFree(dx);cudaFree(dw);cudaFree(doff);cudaFree(dmsk);cudaFree(db);cudaFree(dy);cudaFree(dcols);cudaFree(dwf);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return 0;
}
