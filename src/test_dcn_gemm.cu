// test_dcn_gemm.cu —— 独立验证快速路径(im2col + cuBLAS GEMM)的数值正确性,与 torchvision oracle 比对。
//   先单独验证该路径,再集成进插件,以隔离 kernel 数值与插件接线两类问题。
//
// 编译运行(在 python/ 目录,.bin 位于此处;需链接 cuBLAS):
//   nvcc -O2 -std=c++17 -arch=sm_89 ../src/test_dcn_gemm.cu ../src/dcn_kernel.cu -I../src -lcublas -o test_dcn_gemm
//   ./test_dcn_gemm

#include "dcn_kernel.h"    // deform_im2col_launch / add_bias_launch 声明
#include <cublas_v2.h>     // cuBLAS(cublasSgemm)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){                 \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)  // CUDA 错误检查宏

// 从裸 .bin 读取 n 个 float。
static std::vector<float> load(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "无法打开 %s\n", path); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "%s 大小不符\n", path); exit(1); }  // 个数不符即报错
    fclose(f);
    return v;
}

int main() {
    const int Cin=16, Cout=16, H=32, W=32, K=3, stride=1, pad=1, dil=1, Ho=32, Wo=32;  // 形状/超参(同 01;N=1,快速路径仅支持单 batch)
    const int CKK = Cin*K*K, HW = Ho*Wo;               // CKK=im2col 行数(=weight 列数);HW=输出空间点数(=矩阵列数)
    auto x   = load("dcn_x.bin",      (size_t)Cin*H*W);    // 输入(无 n 维,N=1)
    auto off = load("dcn_offset.bin", (size_t)2*K*K*Ho*Wo);// offset
    auto msk = load("dcn_mask.bin",   (size_t)K*K*Ho*Wo);  // mask
    auto w   = load("dcn_weight.bin", (size_t)Cout*Cin*K*K);   // weight,视作 [Cout, CKK]
    auto b   = load("dcn_bias.bin",   (size_t)Cout);       // bias
    auto yref= load("dcn_y.bin",      (size_t)Cout*Ho*Wo);     // oracle 输出,视作 [Cout, HW]

    float *dx,*doff,*dmsk,*dw,*db,*dy,*dcols;          // device 指针;dcols 为 im2col 中间矩阵
    CK(cudaMalloc(&dx, x.size()*4));    CK(cudaMalloc(&doff, off.size()*4));   // *4 = *sizeof(float)
    CK(cudaMalloc(&dmsk, msk.size()*4)); CK(cudaMalloc(&dw, w.size()*4));
    CK(cudaMalloc(&db, b.size()*4));    CK(cudaMalloc(&dy, yref.size()*4));
    CK(cudaMalloc(&dcols, (size_t)CKK*HW*4));          // cols 矩阵 [CKK, HW]
    CK(cudaMemcpy(dx, x.data(), x.size()*4, cudaMemcpyHostToDevice));   // H2D 全部输入
    CK(cudaMemcpy(doff,off.data(),off.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmsk,msk.data(),msk.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, w.data(), w.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), b.size()*4, cudaMemcpyHostToDevice));

    // 1) im2col:每个采样值计算一次,写入 cols[CKK, HW]。
    deform_im2col_launch(dx, doff, dmsk, dcols, Cin, H, W, Ho, Wo, K, stride, pad, dil, 0);

    // 2) GEMM:out[Cout,HW] = weight[Cout,CKK] @ cols[CKK,HW]。
    //   列主序 cuBLAS 计算行主序 C=A@B 的标准写法:sgemm(N,N, HW,Cout,CKK, cols(ld HW), weight(ld CKK), out(ld HW))。
    cublasHandle_t h; cublasCreate(&h);
    float one=1.f, zero=0.f;                           // GEMM 系数 alpha=1, beta=0(即 C = A@B)
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, HW, Cout, CKK,   // m=HW, n=Cout, k=CKK;均不转置
                &one, dcols, HW, dw, CKK, &zero, dy, HW);     // 利用"行主序 = 列主序转置"等价,直接得到行主序 [Cout,HW]

    // 3) 加 bias。
    add_bias_launch(dy, db, Cout, HW, 0);              // GEMM 不含 bias,单独广播加 bias[oc]
    CK(cudaDeviceSynchronize());

    std::vector<float> y(yref.size());
    CK(cudaMemcpy(y.data(), dy, yref.size()*4, cudaMemcpyDeviceToHost));  // D2H 取回结果
    double maxerr=0; for (size_t i=0;i<y.size();++i) maxerr=fmax(maxerr,fabs((double)y[i]-yref[i]));  // 最大绝对误差 vs oracle
    printf("im2col + cuBLAS GEMM vs torchvision oracle: max|err|=%.3e\n", maxerr);
    printf("%s\n", maxerr<1e-3 ? "[PASS] 快速路径数值正确"
                               : "[FAIL] 数值不匹配:检查 im2col 的 r/pos 索引或 GEMM 的行/列主序参数");
    cublasDestroy(h);
    cudaFree(dx);cudaFree(doff);cudaFree(dmsk);cudaFree(dw);cudaFree(db);cudaFree(dy);cudaFree(dcols);  // 释放显存
    return 0;
}
