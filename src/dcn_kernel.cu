// dcn_kernel.cu —— DCNv2 前向 kernel(模板化,FP32 / FP16 共用同一逻辑,内部统一以 float 累加)。
//
//   FP16 仅用于 I/O 以节省显存与带宽,累加仍走 float 以避免半精度累加误差堆积(标准做法)。
//   计算三步:① 采样位置 = 规则网格位置 + offset;② 双线性采样;③ 乘 mask、乘 weight 后累加。
//   双线性插值的越界处理与 torchvision.ops.deform_conv2d 对齐(逐角点判越界,界外取 0)。

#include "dcn_kernel.h"      // 对外 launcher 声明(dcnv2_launch / _half / _int8 / im2col / add_bias)

// 读取辅助:将不同输入元素类型统一转为 float 参与计算(按实参类型重载)。
__device__ __forceinline__ float to_f(float v) { return v; }                 // float 透传
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }  // half -> float
__device__ __forceinline__ float to_f(signed char v) { return (float)v; }    // INT8 原始整数 -> float(尚未乘 scale)
// 写回辅助:将 float 结果转为目标元素类型 T(主模板 + 显式特化)。
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }

// 双线性插值:在(可能含小数的)采样坐标 (h, w) 处对单通道图 in[H,W] 取值,语义对齐 torchvision。
template <typename T>
__device__ __forceinline__ float bilinear(const T* in, int H, int W, float h, float w) {
    if (h <= -1.f || h >= (float)H || w <= -1.f || w >= (float)W) return 0.f;  // 采样中心落在 [-1,H]x[-1,W] 外则取 0
    int h_low = (int)floorf(h), w_low = (int)floorf(w);   // 左上角整数坐标(向下取整)
    int h_high = h_low + 1, w_high = w_low + 1;            // 右下角整数坐标
    float lh = h - h_low, lw = w - w_low, hh = 1.f - lh, hw = 1.f - lw;  // 到 low/high 的小数距离,即四角权重
    float v1 = (h_low >= 0 && w_low >= 0)           ? to_f(in[h_low * W + w_low])  : 0.f;  // 左上角(逐角点判越界,界外取 0)
    float v2 = (h_low >= 0 && w_high <= W - 1)      ? to_f(in[h_low * W + w_high]) : 0.f;  // 右上角
    float v3 = (h_high <= H - 1 && w_low >= 0)      ? to_f(in[h_high * W + w_low]) : 0.f;  // 左下角
    float v4 = (h_high <= H - 1 && w_high <= W - 1) ? to_f(in[h_high * W + w_high]): 0.f;  // 右下角
    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;   // 四角按面积加权 = 双线性结果
}

// DCNv2 朴素前向 kernel(FP32/FP16 共用):一个线程计算一个输出元素 y[n,oc,oh,ow]。
template <typename T>
__global__ void dcnv2_kernel(const T* __restrict__ x, const T* __restrict__ offset,   // __restrict__:声明指针不别名,允许走只读缓存
                             const T* __restrict__ mask, const T* __restrict__ weight,
                             const T* __restrict__ bias, T* y,
                             int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                             int K, int stride, int pad, int dil) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程号
    int total = N * Cout * Ho * Wo;                     // 输出元素总数
    if (idx >= total) return;                           // 越界线程退出(总数非块大小整数倍时)
    int ow = idx % Wo, oh = (idx / Wo) % Ho, oc = (idx / (Wo * Ho)) % Cout, n = idx / (Wo * Ho * Cout);  // 行主序解包为 (n,oc,oh,ow)
    const int OFF = 2 * K * K, MSK = K * K;             // OFF=offset 通道数(每点 2 分量);MSK=mask 通道数(每点 1 系数)
    int base = oh * Wo + ow;                            // offset/mask 上的空间偏移(其空间尺寸为输出 Ho x Wo)
    float acc = bias ? to_f(bias[oc]) : 0.f;            // 累加器初值为该输出通道的 bias(bias 可为空)

    // offset/mask 与采样坐标仅依赖 (i,j) 而与 ic 无关,故提到 ic 循环外,
    // 避免 Cin 倍的冗余 offset/mask 读取;配合 __restrict__ 走只读缓存。
    for (int i = 0; i < K; ++i) {                       // 遍历卷积窗口行 i
        for (int j = 0; j < K; ++j) {                   // 遍历卷积窗口列 j
            int p = i * K + j;                          // 窗口内采样点序号
            float dh = to_f(offset[((size_t)(n * OFF + 2 * p)     * Ho) * Wo + base]);  // 纵向偏移(offset 通道 2p)
            float dw = to_f(offset[((size_t)(n * OFF + 2 * p + 1) * Ho) * Wo + base]);  // 横向偏移(offset 通道 2p+1)
            float m  = to_f(mask  [((size_t)(n * MSK + p)         * Ho) * Wo + base]);  // 调制系数(mask 通道 p)
            float h_im = oh * stride - pad + i * dil + dh;   // 实际采样行 = 规则网格行 + 偏移 dh
            float w_im = ow * stride - pad + j * dil + dw;   // 实际采样列 = 规则网格列 + 偏移 dw
            for (int ic = 0; ic < Cin; ++ic) {           // 逐输入通道采样(坐标共用,各通道像素值不同)
                const T* xp = x + ((size_t)(n * Cin + ic) * H) * W;   // 第 (n,ic) 张输入特征图起始地址
                float v = bilinear<T>(xp, H, W, h_im, w_im);          // 在 (h_im,w_im) 双线性采样该通道
                acc += to_f(weight[((size_t)(oc * Cin + ic) * K + i) * K + j]) * v * m;  // 累加:权重 x 采样值 x 调制系数
            }
        }
    }
    y[idx] = from_f<T>(acc);   // float 累加结果写回为目标类型 T
}

// 模板 launcher:配置 grid/block 并启动朴素 kernel(供 float / half 复用)。
template <typename T>
static void launch(const T* x, const T* o, const T* m, const T* w, const T* b, T* y,
                   int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                   int K, int s, int p, int d, cudaStream_t st) {
    int total = N * Cout * Ho * Wo, th = 256, bl = (total + th - 1) / th;   // 每块 256 线程;块数向上取整覆盖全部输出
    dcnv2_kernel<T><<<bl, th, 0, st>>>(x, o, m, w, b, y, N, Cin, Cout, H, W, Ho, Wo, K, s, p, d);
}

// 对外入口:FP32 朴素前向。
void dcnv2_launch(const float* x, const float* o, const float* m, const float* w, const float* b,
                  float* y, int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                  int K, int s, int p, int d, cudaStream_t st) {
    launch<float>(x, o, m, w, b, y, N, Cin, Cout, H, W, Ho, Wo, K, s, p, d, st);
}

// 对外入口:FP16 朴素前向(I/O 为 half,内部以 float 累加)。
void dcnv2_launch_half(const __half* x, const __half* o, const __half* m, const __half* w,
                       const __half* b, __half* y, int N, int Cin, int Cout, int H, int W, int Ho,
                       int Wo, int K, int s, int p, int d, cudaStream_t st) {
    launch<__half>(x, o, m, w, b, y, N, Cin, Cout, H, W, Ho, Wo, K, s, p, d, st);
}

// INT8 路径:x/weight 为 INT8(各带 scale 反量化),offset/mask/bias 保持 FP32,输出 FP32。
// 一个线程一个输出元素;此路径未做上述 hoist 优化(ic 置于最外层),为朴素实现。
__global__ void dcnv2_kernel_int8(const int8_t* x, float x_scale,            // x:INT8 激活 + per-tensor 反量化 scale
                                  const float* offset, const float* mask,    // 几何量 offset/mask 不量化,保 FP32
                                  const int8_t* weight, float w_scale, const float* bias, float* y,  // weight:INT8 + scale;bias/输出 FP32
                                  int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                                  int K, int stride, int pad, int dil) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程号
    int total = N * Cout * Ho * Wo;                     // 输出元素总数
    if (idx >= total) return;                           // 越界线程退出
    int ow = idx % Wo, oh = (idx / Wo) % Ho, oc = (idx / (Wo * Ho)) % Cout, n = idx / (Wo * Ho * Cout);  // 解包 (n,oc,oh,ow)
    const int OFF = 2 * K * K, MSK = K * K;             // offset/mask 通道数
    int base = oh * Wo + ow;                            // offset/mask 上的空间偏移
    float acc = bias ? bias[oc] : 0.f;                  // 累加器初值为 bias(FP32)

    for (int ic = 0; ic < Cin; ++ic) {                  // 逐输入通道(本版 ic 置于最外层)
        const int8_t* xp = x + ((size_t)(n * Cin + ic) * H) * W;   // 第 (n,ic) 张 INT8 输入图起始地址
        for (int i = 0; i < K; ++i) {                   // 卷积窗口行
            for (int j = 0; j < K; ++j) {               // 卷积窗口列
                int p = i * K + j;                      // 窗口内采样点序号
                float dh = offset[((size_t)(n * OFF + 2 * p)     * Ho) * Wo + base];   // 纵向偏移(FP32)
                float dw = offset[((size_t)(n * OFF + 2 * p + 1) * Ho) * Wo + base];   // 横向偏移(FP32)
                float mm = mask  [((size_t)(n * MSK + p)         * Ho) * Wo + base];   // 调制系数(FP32)
                float h_im = oh * stride - pad + i * dil + dh;   // 实际采样行
                float w_im = ow * stride - pad + j * dil + dw;   // 实际采样列
                float v = bilinear<int8_t>(xp, H, W, h_im, w_im) * x_scale;            // 整数域双线性采样后乘 x_scale 反量化激活
                float wq = (float)weight[((size_t)(oc * Cin + ic) * K + i) * K + j] * w_scale;  // 权重整数乘 w_scale 反量化
                acc += wq * v * mm;                     // 累加:反量化权重 x 反量化采样值 x 调制
            }
        }
    }
    y[idx] = acc;   // 输出 FP32
}

// 对外入口:INT8 混合精度前向。
void dcnv2_launch_int8(const int8_t* x, float x_scale, const float* offset, const float* mask,
                       const int8_t* weight, float w_scale, const float* bias, float* y,
                       int N, int Cin, int Cout, int H, int W, int Ho, int Wo,
                       int K, int stride, int pad, int dil, cudaStream_t stream) {
    int total = N * Cout * Ho * Wo, th = 256, bl = (total + th - 1) / th;
    dcnv2_kernel_int8<<<bl, th, 0, stream>>>(x, x_scale, offset, mask, weight, w_scale, bias, y,
                                             N, Cin, Cout, H, W, Ho, Wo, K, stride, pad, dil);
}

// ============================================================================
// 快速路径:deformable im2col(FP32,N=1)。每个采样值仅计算一次并写入 cols[Cin*K*K, Ho*Wo]。
//   一个线程一个 cols 元素 (r, pos):r = ic*K*K + i*K + j(与 weight 列序一致),pos = oh*Wo + ow。
// ============================================================================
__global__ void deform_im2col_kernel(const float* __restrict__ x, const float* __restrict__ offset,
                                     const float* __restrict__ mask, float* cols,
                                     int Cin, int H, int W, int Ho, int Wo,
                                     int K, int stride, int pad, int dil) {
    int HW = Ho * Wo, CKK = Cin * K * K;                // HW=输出空间点数(矩阵列数);CKK=im2col 行数(=weight 列数)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // 全局线程号
    if (idx >= CKK * HW) return;                         // 越界线程退出
    int pos = idx % HW, r = idx / HW;                    // 拆为 cols 的 (行 r, 列 pos)
    int j = r % K, i = (r / K) % K, ic = r / (K * K);    // 行 r 拆回 (ic, i, j),与 weight 列布局一致
    int oh = pos / Wo, ow = pos % Wo;                    // 列 pos 拆回输出空间坐标 (oh, ow)
    int p = i * K + j, base = oh * Wo + ow;              // p=窗口点序号;base=offset/mask 空间偏移
    float dh = offset[((size_t)(2 * p)     * Ho) * Wo + base];   // 纵向偏移(N=1,无 n 维偏移)
    float dw = offset[((size_t)(2 * p + 1) * Ho) * Wo + base];   // 横向偏移
    float m  = mask  [((size_t)p          * Ho) * Wo + base];    // 调制系数
    float h_im = oh * stride - pad + i * dil + dh;       // 实际采样行
    float w_im = ow * stride - pad + j * dil + dw;       // 实际采样列
    const float* xp = x + ((size_t)ic * H) * W;          // 第 ic 张输入图起始地址(N=1)
    cols[(size_t)r * HW + pos] = m * bilinear<float>(xp, H, W, h_im, w_im);   // 调制后采样值写入 cols[r,pos];每元素仅算一次
}

// 对外入口:启动 im2col(总线程数 = cols 元素数)。
void deform_im2col_launch(const float* x, const float* offset, const float* mask, float* cols,
                          int Cin, int H, int W, int Ho, int Wo,
                          int K, int stride, int pad, int dil, cudaStream_t stream) {
    int total = Cin * K * K * Ho * Wo, th = 256, bl = (total + th - 1) / th;   // 一个线程填一个 cols 元素
    deform_im2col_kernel<<<bl, th, 0, stream>>>(x, offset, mask, cols,
                                                Cin, H, W, Ho, Wo, K, stride, pad, dil);
}

// 偏置广播:out[oc, pos] += bias[oc]。cuBLAS GEMM 仅计算 weight@cols 不含 bias,故单独补一个 kernel。
__global__ void add_bias_kernel(float* out, const float* __restrict__ bias, int Cout, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程号(覆盖 out 的 Cout*HW 个元素)
    if (idx >= Cout * HW) return;                       // 越界线程退出
    out[idx] += bias[idx / HW];                         // out 行主序 [Cout,HW],idx/HW 即输出通道 oc
}

// 对外入口:启动偏置广播。
void add_bias_launch(float* out, const float* bias, int Cout, int HW, cudaStream_t stream) {
    int total = Cout * HW, th = 256, bl = (total + th - 1) / th;   // 一个线程加一个输出元素
    add_bias_kernel<<<bl, th, 0, stream>>>(out, bias, Cout, HW);
}

// ============================================================================
// FP16 / INT8 快速路径(im2col + cuBLAS GEMM),与 FP32 快速路径同构。
// ============================================================================
// FP16:cols 为 __half,供 cublasGemmEx。
__global__ void deform_im2col_half_kernel(const __half* __restrict__ x, const __half* __restrict__ offset,
                                          const __half* __restrict__ mask, __half* cols,
                                          int Cin, int H, int W, int Ho, int Wo, int K, int stride, int pad, int dil) {
    int HW = Ho * Wo, CKK = Cin * K * K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= CKK * HW) return;
    int pos = idx % HW, r = idx / HW;
    int j = r % K, i = (r / K) % K, ic = r / (K * K);
    int oh = pos / Wo, ow = pos % Wo;
    int p = i * K + j, base = oh * Wo + ow;
    float dh = to_f(offset[((size_t)(2 * p)     * Ho) * Wo + base]);
    float dw = to_f(offset[((size_t)(2 * p + 1) * Ho) * Wo + base]);
    float m  = to_f(mask  [((size_t)p          * Ho) * Wo + base]);
    float h_im = oh * stride - pad + i * dil + dh;
    float w_im = ow * stride - pad + j * dil + dw;
    const __half* xp = x + ((size_t)ic * H) * W;
    cols[(size_t)r * HW + pos] = __float2half(m * bilinear<__half>(xp, H, W, h_im, w_im));
}
void deform_im2col_half_launch(const __half* x, const __half* offset, const __half* mask, __half* cols,
                               int Cin, int H, int W, int Ho, int Wo,
                               int K, int stride, int pad, int dil, cudaStream_t stream) {
    int total = Cin * K * K * Ho * Wo, th = 256, bl = (total + th - 1) / th;
    deform_im2col_half_kernel<<<bl, th, 0, stream>>>(x, offset, mask, cols, Cin, H, W, Ho, Wo, K, stride, pad, dil);
}
__global__ void add_bias_half_kernel(__half* out, const __half* __restrict__ bias, int Cout, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * HW) return;
    out[idx] = __float2half(__half2float(out[idx]) + __half2float(bias[idx / HW]));
}
void add_bias_half_launch(__half* out, const __half* bias, int Cout, int HW, cudaStream_t stream) {
    int total = Cout * HW, th = 256, bl = (total + th - 1) / th;
    add_bias_half_kernel<<<bl, th, 0, stream>>>(out, bias, Cout, HW);
}
// INT8:cols 为 float = m * bilinear(x_int8)(整数域采样,scale 留到最后)。
__global__ void deform_im2col_int8_kernel(const int8_t* __restrict__ x, const float* __restrict__ offset,
                                          const float* __restrict__ mask, float* cols,
                                          int Cin, int H, int W, int Ho, int Wo, int K, int stride, int pad, int dil) {
    int HW = Ho * Wo, CKK = Cin * K * K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= CKK * HW) return;
    int pos = idx % HW, r = idx / HW;
    int j = r % K, i = (r / K) % K, ic = r / (K * K);
    int oh = pos / Wo, ow = pos % Wo;
    int p = i * K + j, base = oh * Wo + ow;
    float dh = offset[((size_t)(2 * p)     * Ho) * Wo + base];
    float dw = offset[((size_t)(2 * p + 1) * Ho) * Wo + base];
    float m  = mask  [((size_t)p          * Ho) * Wo + base];
    float h_im = oh * stride - pad + i * dil + dh;
    float w_im = ow * stride - pad + j * dil + dw;
    const int8_t* xp = x + ((size_t)ic * H) * W;
    cols[(size_t)r * HW + pos] = m * bilinear<int8_t>(xp, H, W, h_im, w_im);
}
void deform_im2col_int8_launch(const int8_t* x, const float* offset, const float* mask, float* cols,
                               int Cin, int H, int W, int Ho, int Wo, int K, int stride, int pad, int dil, cudaStream_t stream) {
    int total = Cin * K * K * Ho * Wo, th = 256, bl = (total + th - 1) / th;
    deform_im2col_int8_kernel<<<bl, th, 0, stream>>>(x, offset, mask, cols, Cin, H, W, Ho, Wo, K, stride, pad, dil);
}
__global__ void dequant_weight_kernel(const int8_t* __restrict__ w, float* wf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) wf[idx] = (float)w[idx];
}
void dequant_weight_launch(const int8_t* w, float* wf, int n, cudaStream_t stream) {
    int th = 256, bl = (n + th - 1) / th;
    dequant_weight_kernel<<<bl, th, 0, stream>>>(w, wf, n);
}
__global__ void add_bias_scale_kernel(float* out, const float* __restrict__ bias, float scale, int Cout, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * HW) return;
    out[idx] = out[idx] * scale + bias[idx / HW];
}
void add_bias_scale_launch(float* out, const float* bias, float scale, int Cout, int HW, cudaStream_t stream) {
    int total = Cout * HW, th = 256, bl = (total + th - 1) / th;
    add_bias_scale_kernel<<<bl, th, 0, stream>>>(out, bias, scale, Cout, HW);
}
