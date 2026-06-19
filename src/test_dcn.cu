// test_dcn.cu —— DCNv2 kernel 的独立单元测试:读取 oracle 落盘的 .bin,运行 kernel,与参考输出比对。
//   分层验证策略:先单独验证 kernel 数值正确性,再集成进插件。
//
// 编译运行(在 python/ 目录,.bin 位于此处):
//   cd ../python   # 需先运行 01_oracle_and_export.py 生成 dcn_*.bin
//   nvcc -O2 -std=c++17 -arch=sm_89 ../src/test_dcn.cu ../src/dcn_kernel.cu -I../src -o test_dcn
//   ./test_dcn

#include "dcn_kernel.h"   // 被测对象:dcnv2_launch 等 launcher 声明
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// CUDA 错误检查宏:出错时打印 文件:行 与错误信息并退出。
#define CK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){                 \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// 从裸 .bin 读取 n 个 float 到 vector(对应 01 中 .tofile 落盘的文件)。
static std::vector<float> load(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "无法打开 %s(请先在 python/ 运行 01_oracle_and_export.py)\n", path); exit(1); }
    std::vector<float> v(n);
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s 大小不符:读到 %zu,期望 %zu\n", path, got, n); exit(1); }  // 个数不符即形状不匹配
    return v;
}

int main() {
    // 形状与超参须与 01_oracle_and_export.py 完全一致,否则索引与比对均会出错。
    const int N=1, Cin=16, Cout=16, H=32, W=32, K=3, stride=1, pad=1, dil=1;
    const int Ho=32, Wo=32;                            // 输出空间尺寸(stride=1,pad=1,K=3 时与输入一致)

    auto x   = load("dcn_x.bin",      (size_t)N*Cin*H*W);     // 输入 x [1,16,32,32]
    auto off = load("dcn_offset.bin", (size_t)N*2*K*K*Ho*Wo); // offset [1,18,32,32]
    auto msk = load("dcn_mask.bin",   (size_t)N*K*K*Ho*Wo);   // mask [1,9,32,32]
    auto w   = load("dcn_weight.bin", (size_t)Cout*Cin*K*K);  // weight [16,16,3,3]
    auto b   = load("dcn_bias.bin",   (size_t)Cout);          // bias [16]
    auto yref= load("dcn_y.bin",      (size_t)N*Cout*Ho*Wo);  // oracle 输出 y [1,16,32,32](参考答案)

    float *dx,*doff,*dmsk,*dw,*db,*dy;                 // 6 个 device 指针
    CK(cudaMalloc(&dx,  x.size()  *sizeof(float)));    // 在 GPU 上为各张量分配显存
    CK(cudaMalloc(&doff,off.size()*sizeof(float)));
    CK(cudaMalloc(&dmsk,msk.size()*sizeof(float)));
    CK(cudaMalloc(&dw,  w.size()  *sizeof(float)));
    CK(cudaMalloc(&db,  b.size()  *sizeof(float)));
    CK(cudaMalloc(&dy,  yref.size()*sizeof(float)));   // 输出缓冲(大小同 oracle)
    CK(cudaMemcpy(dx,  x.data(),  x.size()  *sizeof(float), cudaMemcpyHostToDevice));  // H2D:输入拷至 GPU
    CK(cudaMemcpy(doff,off.data(),off.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmsk,msk.data(),msk.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw,  w.data(),  w.size()  *sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,  b.data(),  b.size()  *sizeof(float), cudaMemcpyHostToDevice));

    dcnv2_launch(dx,doff,dmsk,dw,db,dy, N,Cin,Cout,H,W,Ho,Wo,K,stride,pad,dil, 0);  // 运行 FP32 朴素 kernel(默认流)
    CK(cudaGetLastError());                            // 检查 kernel 启动是否出错
    CK(cudaDeviceSynchronize());                       // 等待 GPU 执行完成

    std::vector<float> y(yref.size());                 // 主机端结果缓冲
    CK(cudaMemcpy(y.data(), dy, yref.size()*sizeof(float), cudaMemcpyDeviceToHost));  // D2H:取回结果

    double maxerr=0.0, sum=0.0;                         // 统计最大绝对误差与结果总和
    for (size_t i=0;i<y.size();++i){ maxerr=fmax(maxerr, fabs((double)y[i]-yref[i])); sum+=y[i]; }
    printf("DCN kernel vs torchvision oracle: max|err|=%.3e  (y.sum=%.4f, oracle=-30.9894)\n",  // 总和可作快速校验
           maxerr, sum);
    printf("%s\n", maxerr < 1e-3                        // 误差阈值判定(1e-3)
           ? "[PASS] kernel 数值与 torchvision 对齐"
           : "[FAIL] 数值不匹配:检查 offset/mask 通道索引、bilinear 边界或 h_im/w_im 公式");

    cudaFree(dx);cudaFree(doff);cudaFree(dmsk);cudaFree(dw);cudaFree(db);cudaFree(dy);  // 释放显存
    return 0;
}
