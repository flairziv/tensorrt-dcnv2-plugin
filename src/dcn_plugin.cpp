// dcn_plugin.cpp —— DCNv2 的 TensorRT 10 IPluginV3 插件(含 Creator 与注册宏)。
//
// 标准 IPluginV3 结构(Core / Build / Runtime + Creator + 注册宏)。DCN 特有之处:
//   1. 5 个输入:input, offset, mask, weight, bias(weight/bias 在 ONNX 中为常量 initializer);
//   2. 输出形状不同于输入:Ho = (H + 2*pad - dil*(K-1) - 1)/stride + 1,经 IExprBuilder 表达;
//   3. 读取 5 个属性:stride / padding / dilation / kernel / deformable_groups(名称与 ONNX 节点一致)。
// enqueue 仅调用经独立单元测试验证的 kernel launcher。

#include "dcn_kernel.h"        // kernel launcher(dcnv2_launch / _half / _int8 / im2col / add_bias)

#include <NvInfer.h>           // TensorRT 核心接口(IPluginV3 等)
#include <NvInferRuntime.h>    // 运行时接口(PluginTensorDesc / IPluginResourceContext 等)
#include <cublas_v2.h>         // cuBLAS(FP32 快速路径的 GEMM)

#include <cstring>             // std::strcmp(比对属性名)
#include <string>              // std::string(namespace 字段)
#include <vector>              // std::vector(序列化字段集合)

using namespace nvinfer1;

namespace {
constexpr char const* kNAME = "DCNv2";    // 算子名,须与 ONNX 节点 op_type 及 Creator 保持一致
constexpr char const* kVERSION = "1";     // 版本号,须与 Creator 保持一致
}  // namespace

// ============================================================================
// 插件本体
// ============================================================================
// 多重继承 IPluginV3 的能力接口:Core(身份)/ Build(编译期)/ Runtime(执行期)。
class DCNv2Plugin : public IPluginV3,
                    public IPluginV3OneCore,
                    public IPluginV3OneBuild,
                    public IPluginV3OneRuntime {
public:
    // 构造:保存 5 个卷积超参(由 Creator 从 ONNX 属性解析后传入)。
    DCNv2Plugin(int kernel, int stride, int pad, int dil, int defGroups)
        : mKernel(kernel), mStride(stride), mPad(pad), mDil(dil), mDefGroups(defGroups) {}
    ~DCNv2Plugin() { if (mHandle) cublasDestroy(mHandle); }   // 析构:释放惰性创建的 cuBLAS 句柄

    // ---------- IPluginV3 ----------
    // 按需返回某一类能力的接口指针(同一 this 转为对应基类)。
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
            case PluginCapabilityType::kBUILD:   return static_cast<IPluginV3OneBuild*>(this);    // 编译期能力
            case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);  // 执行期能力
            case PluginCapabilityType::kCORE:    return static_cast<IPluginV3OneCore*>(this);     // 核心身份
        }
        return nullptr;
    }
    // 克隆自身(TRT 在构建及不同上下文中会复制插件),仅需复制超参。
    IPluginV3* clone() noexcept override {
        return new DCNv2Plugin(mKernel, mStride, mPad, mDil, mDefGroups);
    }

    // ---------- Core ----------(身份:名称 / 版本 / 命名空间,须与 Creator 一致)
    AsciiChar const* getPluginName()      const noexcept override { return kNAME; }
    AsciiChar const* getPluginVersion()   const noexcept override { return kVERSION; }
    AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    // ---------- Build ----------(编译期 TRT 的查询接口)
    int32_t getNbOutputs() const noexcept override { return 1; }   // 输出张量个数

    // 输出 dtype 跟随输入 x。
    int32_t getOutputDataTypes(DataType* outTypes, int32_t /*nbOut*/,
                               DataType const* inTypes, int32_t /*nbIn*/) const noexcept override {
        outTypes[0] = inTypes[0];
        return 0;
    }

    // 输出形状 [N, Cout, Ho, Wo]:Cout 取自 weight,Ho/Wo 由卷积公式得出。
    // 形状以符号表达式表示以支持动态维,故使用 IExprBuilder 构造而非直接整数运算。
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t /*nbIn*/,
                            DimsExprs const* /*shapeIn*/, int32_t /*nbShapeIn*/,
                            DimsExprs* outputs, int32_t /*nbOut*/,
                            IExprBuilder& eb) noexcept override {
        // inputs[0]=x[N,Cin,H,W],inputs[3]=weight[Cout,Cin,K,K]
        const int c = 2 * mPad - mDil * (mKernel - 1) - 1;     // 卷积输出公式的常数项:2*pad - dil*(K-1) - 1
        auto outDim = [&](IDimensionExpr const* in) {          // 由输入尺寸 in 计算输出尺寸 (in + c)/stride + 1
            auto t = eb.operation(DimensionOperation::kSUM, *in, *eb.constant(c));             // t = in + c
            auto d = eb.operation(DimensionOperation::kFLOOR_DIV, *t, *eb.constant(mStride));  // d = floor(t / stride)
            return eb.operation(DimensionOperation::kSUM, *d, *eb.constant(1));                // d + 1
        };
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];          // N(同输入 batch)
        outputs[0].d[1] = inputs[3].d[0];          // Cout(weight 的输出通道)
        outputs[0].d[2] = outDim(inputs[0].d[2]);  // Ho
        outputs[0].d[3] = outDim(inputs[0].d[3]);  // Wo
        return 0;
    }

    // 6 个位置(5 输入 + 1 输出,线性布局),支持三种精度组合:
    //   全 FP32 / 全 FP16 / 混合 INT8(x 与 weight 为 INT8,offset/mask/bias/output 为 FP32)。
    // TRT 逐 (位置, 类型) 组合查询是否支持;仅返回 true 的组合会被选用。
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t /*nbIn*/, int32_t /*nbOut*/) noexcept override {
        if (inOut[pos].desc.format != TensorFormat::kLINEAR) return false;  // 仅支持线性(NCHW 连续)布局
        DataType t = inOut[pos].desc.type;          // 当前查询位置的候选类型
        if (pos == 0)                              // 位置 0(x):允许 FP32 / FP16 / INT8
            return t == DataType::kFLOAT || t == DataType::kHALF || t == DataType::kINT8;
        DataType t0 = inOut[0].desc.type;          // 其余位置的类型需与 x 的选择协调
        if (t0 == DataType::kINT8)                 // x 为 INT8(混合精度):仅 weight(pos3)为 INT8,其余 FP32
            return (pos == 3) ? (t == DataType::kINT8) : (t == DataType::kFLOAT);
        return t == t0;                            // x 为 FP32 或 FP16:所有位置与 x 一致
    }
    // 配置回调:本插件无需在此预处理。
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                            DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }

    // 快速路径(FP32)所需 scratch:cols[Cin*K*K, Ho*Wo](float),TRT 据此预留 workspace。
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t /*nbIn*/,
                            DynamicPluginTensorDesc const* out, int32_t /*nbOut*/) const noexcept override {
        int Cin = in[0].desc.dims.d[1];                             // 由输入 x 取 Cin
        int Ho = out[0].desc.dims.d[2], Wo = out[0].desc.dims.d[3]; // 由输出取 Ho/Wo
        return (size_t)Cin * mKernel * mKernel * Ho * Wo * sizeof(float);  // im2col 中间矩阵字节数
    }

    // ---------- Runtime ----------(执行期)
    // enqueue:在给定 stream 上执行本层。inputs/outputs 为已绑定的 device 指针,inDesc/outDesc 提供形状、类型与 scale。
    int32_t enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const* outDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override {
        const auto& xd = inDesc[0].dims;     // x[N,Cin,H,W]
        const auto& wd = inDesc[3].dims;     // weight[Cout,Cin,K,K]
        const auto& yd = outDesc[0].dims;    // y[N,Cout,Ho,Wo]
        int N = xd.d[0], Cin = xd.d[1], H = xd.d[2], W = xd.d[3];
        int Cout = wd.d[0], Ho = yd.d[2], Wo = yd.d[3];
        // 输入顺序:0=x 1=offset 2=mask 3=weight 4=bias;按 x 的精度分派。
        if (inDesc[0].type == DataType::kINT8) {
            // 混合精度:x/weight 为 INT8(scale 取自 desc.scale),offset/mask/bias 为 FP32,输出 FP32。
            dcnv2_launch_int8(
                static_cast<const int8_t*>(inputs[0]), inDesc[0].scale,       // x 与 x_scale(TRT 校准得到的 per-tensor scale)
                static_cast<const float*>(inputs[1]), static_cast<const float*>(inputs[2]),  // offset / mask(FP32)
                static_cast<const int8_t*>(inputs[3]), inDesc[3].scale,       // weight 与 w_scale
                static_cast<const float*>(inputs[4]), static_cast<float*>(outputs[0]),       // bias / 输出(FP32)
                N, Cin, Cout, H, W, Ho, Wo, mKernel, mStride, mPad, mDil, stream);
        } else if (inDesc[0].type == DataType::kHALF) {
            dcnv2_launch_half(                                               // FP16 路径:走 half 朴素 kernel
                static_cast<const __half*>(inputs[0]), static_cast<const __half*>(inputs[1]),
                static_cast<const __half*>(inputs[2]), static_cast<const __half*>(inputs[3]),
                static_cast<const __half*>(inputs[4]), static_cast<__half*>(outputs[0]),
                N, Cin, Cout, H, W, Ho, Wo, mKernel, mStride, mPad, mDil, stream);
        } else {
            // FP32 快速路径:im2col(每采样计算一次)-> cuBLAS GEMM(weight@cols)-> 加 bias。
            float* cols = static_cast<float*>(workspace);                    // 复用 TRT 提供的 workspace 作为 cols 矩阵
            deform_im2col_launch(static_cast<const float*>(inputs[0]),       // 1) 可变形采样展开为 cols[CKK,HW]
                                 static_cast<const float*>(inputs[1]),
                                 static_cast<const float*>(inputs[2]), cols,
                                 Cin, H, W, Ho, Wo, mKernel, mStride, mPad, mDil, stream);
            if (!mHandle) cublasCreate(&mHandle);                            // 惰性创建 cuBLAS 句柄(首次 enqueue)
            cublasSetStream(mHandle, stream);                                // 绑定到同一 stream 保证执行有序
            int CKK = Cin * mKernel * mKernel, HW = Ho * Wo;                 // 矩阵维度
            float one = 1.f, zero = 0.f;                                     // GEMM 系数 alpha / beta
            // 行主序 out[Cout,HW] = weight[Cout,CKK] @ cols[CKK,HW] 在列主序 cuBLAS 下的标准写法。
            cublasSgemm(mHandle, CUBLAS_OP_N, CUBLAS_OP_N, HW, Cout, CKK, &one,  // 2) GEMM:m=HW, n=Cout, k=CKK
                        cols, HW, static_cast<const float*>(inputs[3]), CKK, &zero,  // A=cols(ld HW),B=weight(ld CKK)
                        static_cast<float*>(outputs[0]), HW);                        // C=输出(ld HW)
            add_bias_launch(static_cast<float*>(outputs[0]),                 // 3) 广播加 bias
                            static_cast<const float*>(inputs[4]), Cout, HW, stream);
        }
        return 0;
    }
    // 形状变化回调:本插件无状态需更新。
    int32_t onShapeChange(PluginTensorDesc const*, int32_t,
                          PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    // 绑定到执行上下文:返回可独立运行的实例(克隆),保证多上下文 / 多流安全。
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    // 声明需序列化进引擎的字段(构建时写入,反序列化时由 Creator 读回以重建插件)。
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        mSerial.clear();
        mSerial.emplace_back("stride", &mStride, PluginFieldType::kINT32, 1);
        mSerial.emplace_back("padding", &mPad, PluginFieldType::kINT32, 1);
        mSerial.emplace_back("dilation", &mDil, PluginFieldType::kINT32, 1);
        mSerial.emplace_back("kernel", &mKernel, PluginFieldType::kINT32, 1);
        mSerial.emplace_back("deformable_groups", &mDefGroups, PluginFieldType::kINT32, 1);
        mFC.nbFields = static_cast<int32_t>(mSerial.size());
        mFC.fields = mSerial.data();
        return &mFC;
    }

private:
    int mKernel, mStride, mPad, mDil, mDefGroups;   // 5 个卷积超参
    cublasHandle_t mHandle{nullptr};                // cuBLAS 句柄(惰性创建)
    std::string mNamespace;                          // 命名空间(默认空)
    std::vector<PluginField> mSerial;                // 序列化字段缓冲
    PluginFieldCollection mFC{};                     // 序列化字段集合(指向 mSerial)
};

// ============================================================================
// Creator 工厂:TRT 通过它创建插件(构建期从 ONNX 属性创建,运行期从序列化字段创建)。
// ============================================================================
class DCNv2Creator : public IPluginCreatorV3One {
public:
    DCNv2Creator() {
        // 声明本算子接受的属性名(供 TRT 校验 ONNX 节点属性);值在 createPlugin 时填入。
        for (const char* nm : {"stride", "padding", "dilation", "kernel", "deformable_groups"})
            mFields.emplace_back(nm, nullptr, PluginFieldType::kINT32, 1);
        mFC.nbFields = static_cast<int32_t>(mFields.size());
        mFC.fields = mFields.data();
    }
    AsciiChar const* getPluginName()      const noexcept override { return kNAME; }     // 须与插件名一致
    AsciiChar const* getPluginVersion()   const noexcept override { return kVERSION; }  // 须与插件版本一致
    AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }     // 返回属性名清单

    // 从字段读取 5 个属性并创建插件(构建期 fc 为 ONNX 属性,运行期 fc 为序列化字段)。
    IPluginV3* createPlugin(AsciiChar const* /*name*/, PluginFieldCollection const* fc,
                            TensorRTPhase /*phase*/) noexcept override {
        int stride = 1, pad = 1, dil = 1, kernel = 3, defGroups = 1;   // 默认值(属性缺失时兜底)
        for (int i = 0; i < fc->nbFields; ++i) {
            const auto& f = fc->fields[i];
            int v = f.data ? *static_cast<const int*>(f.data) : 0;     // 取字段 int 值(无数据则 0)
            if (!std::strcmp(f.name, "stride")) stride = v;
            else if (!std::strcmp(f.name, "padding")) pad = v;
            else if (!std::strcmp(f.name, "dilation")) dil = v;
            else if (!std::strcmp(f.name, "kernel")) kernel = v;
            else if (!std::strcmp(f.name, "deformable_groups")) defGroups = v;
        }
        return new DCNv2Plugin(kernel, stride, pad, dil, defGroups);
    }

private:
    std::string mNamespace;             // 命名空间
    std::vector<PluginField> mFields;   // 属性名清单缓冲
    PluginFieldCollection mFC{};        // 属性集合(指向 mFields)
};

// 注册:加载本动态库时自动将 Creator 注册到全局插件表,ONNX 解析器即可按名称解析 DCNv2 节点。
REGISTER_TENSORRT_PLUGIN(DCNv2Creator);
