// test_dcn_int8.cu —— INT8 DCN kernel 的独立单元测试:读取 FP32 bins,将 x/weight 量化为 int8,
//   运行 int8 kernel 并与 oracle 比对。先单独验证 INT8 数值,再集成进插件 / Q-DQ。
//   方案:仅量化 x 与 weight(per-tensor 对称),offset/mask/bias 保持 FP32,输出 FP32。
//
// 编译运行(在 python/ 目录,.bin 位于此处):
//   nvcc -O2 -std=c++17 -arch=sm_89 ../src/test_dcn_int8.cu ../src/dcn_kernel.cu -I../src -o test_dcn_int8
//   ./test_dcn_int8

#include "dcn_kernel.h"   // dcnv2_launch_int8 声明
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

// per-tensor 对称量化:scale = max|.|/127,四舍五入并截断到 [-127,127]。
// 返回 scale(反量化时乘以 scale);量化整数写入 dst。
static float quantize(const std::vector<float>& src, std::vector<int8_t>& dst) {
    float mx = 0.f;                                    // 张量绝对值最大
    for (float v : src) mx = fmaxf(mx, fabsf(v));
    float scale = (mx > 0.f) ? mx / 127.f : 1.f;       // scale = max|.| / 127;全 0 时兜底为 1 避免除零
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        int q = (int)lroundf(src[i] / scale);          // 除以 scale 后四舍五入
        q = q < -127 ? -127 : (q > 127 ? 127 : q);     // 截断到 [-127,127](对称量化不使用 -128)
        dst[i] = (int8_t)q;
    }
    return scale;
}

int main() {
    const int N=1, Cin=16, Cout=16, H=32, W=32, K=3, stride=1, pad=1, dil=1, Ho=32, Wo=32;  // 形状/超参(同 01)
    auto x   = load("dcn_x.bin",      (size_t)N*Cin*H*W);     // FP32 输入
    auto off = load("dcn_offset.bin", (size_t)N*2*K*K*Ho*Wo); // offset(保持 FP32)
    auto msk = load("dcn_mask.bin",   (size_t)N*K*K*Ho*Wo);   // mask(保持 FP32)
    auto w   = load("dcn_weight.bin", (size_t)Cout*Cin*K*K);  // FP32 权重
    auto b   = load("dcn_bias.bin",   (size_t)Cout);          // bias(保持 FP32)
    auto yref= load("dcn_y.bin",      (size_t)N*Cout*Ho*Wo);  // FP32 oracle 输出

    std::vector<int8_t> xq, wq;                         // 量化后的 x、weight
    float xs = quantize(x, xq), ws = quantize(w, wq);   // 分别量化,得到各自 scale
    printf("量化:x_scale=%.5f  w_scale=%.5f(per-tensor 对称 INT8)\n", xs, ws);

    int8_t *dx,*dw; float *doff,*dmsk,*db,*dy;          // x/weight 为 int8 指针,其余为 float
    CK(cudaMalloc(&dx, xq.size())); CK(cudaMalloc(&dw, wq.size()));      // int8 每元素 1 字节
    CK(cudaMalloc(&doff,off.size()*4)); CK(cudaMalloc(&dmsk,msk.size()*4));
    CK(cudaMalloc(&db, b.size()*4));    CK(cudaMalloc(&dy, yref.size()*4));
    CK(cudaMemcpy(dx, xq.data(), xq.size(), cudaMemcpyHostToDevice));    // H2D int8 输入
    CK(cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(doff,off.data(),off.size()*4,cudaMemcpyHostToDevice)); // H2D FP32 几何量
    CK(cudaMemcpy(dmsk,msk.data(),msk.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(),  b.size()*4,  cudaMemcpyHostToDevice));

    dcnv2_launch_int8(dx, xs, doff, dmsk, dw, ws, db, dy,   // 运行 INT8 kernel:传入 x/weight 及其 scale,kernel 内反量化
                      N,Cin,Cout,H,W,Ho,Wo,K,stride,pad,dil, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<float> y(yref.size());
    CK(cudaMemcpy(y.data(), dy, yref.size()*4, cudaMemcpyDeviceToHost));  // D2H

    double maxerr=0, sumabs=0, sumref=0;                // 统计:最大绝对误差、误差绝对值和、参考绝对值和
    for (size_t i=0;i<y.size();++i){ maxerr=fmax(maxerr,fabs((double)y[i]-yref[i]));
        sumabs+=fabs((double)y[i]-yref[i]); sumref+=fabs((double)yref[i]); }
    double rel = sumabs / (sumref + 1e-12);             // 平均相对误差 = Σ|err| / Σ|ref|
    printf("INT8 DCN vs FP32 oracle: max|err|=%.4f  平均相对误差=%.2f%%\n", maxerr, rel*100);
    printf("%s\n", rel < 0.1                            // 相对误差 <10% 视为合理
        ? "[PASS] INT8 DCN 数值在合理范围(x/weight 量化)"
        : "[WARN] 相对误差偏大,检查量化 scale 或 kernel");

    cudaFree(dx);cudaFree(dw);cudaFree(doff);cudaFree(dmsk);cudaFree(db);cudaFree(dy);  // 释放显存
    return 0;
}
