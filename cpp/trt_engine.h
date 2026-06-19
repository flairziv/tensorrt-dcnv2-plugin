// trt_engine.h —— 面向边缘部署的可复用 TensorRT 引擎封装(纯头文件,RAII)。
//
//   将"加载插件、反序列化引擎、发现 I/O、管理显存、enqueueV3 推理、计时"封装为一个类,
//   应用侧实例化一次即可反复 infer(),无需重复编写样板代码。
//
//   适用范围:本项目的引擎为 backbone(ResNet50 + FPN + DCN),输出 5 张 FPN 特征图(非最终框);
//     检测头与 NMS 当前置于 PyTorch(拆分部署)。因此本类是引擎 runtime,而非直接输出 box 的 detector。
//   假设:I/O 张量为 FP32。若引擎 I/O 为 FP16,getOutput 需另作类型转换。
//   Jetson(ARM)与本机(x86/WSL)的 TensorRT C++ API 一致,本封装可直接迁移至 Jetson。

#pragma once
#include <NvInfer.h>          // TensorRT C++ 核心
#include <cuda_runtime.h>     // CUDA 运行时
#include <dlfcn.h>            // dlopen 加载插件 .so

#include <cstdio>
#include <cstdint>
#include <fstream>            // 读取引擎文件
#include <stdexcept>          // 异常
#include <string>
#include <vector>

namespace dcn {

// CUDA 错误检查:出错抛异常,确保 RAII 下正确析构。
inline void cudaCheck(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(e)
                                 + " @ " + file + ":" + std::to_string(line));
}
#define DCN_CK(call) ::dcn::cudaCheck((call), __FILE__, __LINE__)

// 各 TensorRT 数据类型的字节数(用于计算显存大小)。
inline size_t dtypeSize(nvinfer1::DataType t) {
    using DT = nvinfer1::DataType;
    switch (t) {
        case DT::kFLOAT: return 4;
        case DT::kHALF:  return 2;
        case DT::kINT8:  return 1;
        case DT::kUINT8: return 1;
        case DT::kINT32: return 4;
        case DT::kINT64: return 8;
        case DT::kBOOL:  return 1;
        default:         return 4;     // 兜底按 4 字节
    }
}

// 最小日志器:仅打印 WARNING 及以上。
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};

// 单个 I/O 张量的元信息及其 device 缓冲。
struct IOTensor {
    std::string name;          // 张量名
    nvinfer1::Dims dims;       // 形状
    bool isInput;              // 输入或输出
    nvinfer1::DataType dtype;  // 数据类型
    size_t vol;                // 元素个数(各维相乘)
    size_t bytes;              // 字节数 = vol * dtypeSize
    void* dptr;                // device 显存指针
};

// ============================================================================
// TrtEngine:加载一次,反复推理。
// ============================================================================
class TrtEngine {
public:
    // 构造:加载插件(.so,可空)-> 反序列化引擎 -> 创建上下文 -> 发现并分配所有 I/O。
    explicit TrtEngine(const std::string& enginePath, const std::string& pluginPath = "") {
        if (!pluginPath.empty()) {                                  // 1) 加载插件,注册 DCNv2 Creator
            mPlugin = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!mPlugin) throw std::runtime_error(std::string("dlopen 失败: ") + dlerror());
        }
        std::vector<char> blob = readFile(enginePath);              // 2) 读取引擎字节流
        mRuntime = nvinfer1::createInferRuntime(mLogger);
        mEngine = mRuntime->deserializeCudaEngine(blob.data(), blob.size());  // 反序列化(依赖上面注册的插件)
        if (!mEngine) throw std::runtime_error("反序列化失败(引擎/插件/GPU/TRT 版本不匹配)");
        mCtx = mEngine->createExecutionContext();
        DCN_CK(cudaStreamCreate(&mStream));                        // 创建推理流

        int n = mEngine->getNbIOTensors();                         // 3) 发现所有 I/O 张量
        for (int i = 0; i < n; ++i) {
            IOTensor t;
            t.name = mEngine->getIOTensorName(i);
            t.dims = mEngine->getTensorShape(t.name.c_str());
            t.dtype = mEngine->getTensorDataType(t.name.c_str());
            t.isInput = mEngine->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
            t.vol = 1;
            for (int k = 0; k < t.dims.nbDims; ++k) t.vol *= (t.dims.d[k] > 0 ? t.dims.d[k] : 1);
            t.bytes = t.vol * dtypeSize(t.dtype);
            DCN_CK(cudaMalloc(&t.dptr, t.bytes));                  // 分配显存
            mCtx->setTensorAddress(t.name.c_str(), t.dptr);        // 按名称绑定地址到上下文(TRT10 方式)
            mIO.push_back(t);
        }
    }

    // 析构:逆序释放(显存 -> 上下文 -> 引擎 -> runtime -> 流 -> 插件)。
    ~TrtEngine() {
        for (auto& t : mIO) if (t.dptr) cudaFree(t.dptr);
        delete mCtx;
        delete mEngine;
        delete mRuntime;
        if (mStream) cudaStreamDestroy(mStream);
        if (mPlugin) dlclose(mPlugin);
    }

    TrtEngine(const TrtEngine&) = delete;             // 禁止拷贝(持有裸资源)
    TrtEngine& operator=(const TrtEngine&) = delete;

    // 写入某输入张量(host float -> device)。count 为 0 时按该张量元素数。
    void setInput(const std::string& name, const float* host, size_t count = 0) {
        IOTensor& t = at(name);
        size_t n = count ? count : t.vol;
        DCN_CK(cudaMemcpy(t.dptr, host, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // 执行一次推理(异步入队 + 同步等待)。
    void infer() {
        if (!mCtx->enqueueV3(mStream)) throw std::runtime_error("enqueueV3 失败");
        DCN_CK(cudaStreamSynchronize(mStream));
    }

    // 取某输出张量(device -> host;假设 FP32)。
    std::vector<float> getOutput(const std::string& name) const {
        const IOTensor& t = at(name);
        std::vector<float> host(t.vol);
        DCN_CK(cudaMemcpy(host.data(), t.dptr, t.vol * sizeof(float), cudaMemcpyDeviceToHost));
        return host;
    }

    // 纯 GPU 延迟基准(cudaEvent,先预热):返回单次平均 ms(不含 H2D/D2H)。
    float benchmark(int iters = 200) {
        mCtx->enqueueV3(mStream);                     // 预热(触发算法选择与惰性分配)
        DCN_CK(cudaStreamSynchronize(mStream));
        cudaEvent_t t0, t1;
        DCN_CK(cudaEventCreate(&t0));
        DCN_CK(cudaEventCreate(&t1));
        DCN_CK(cudaEventRecord(t0, mStream));
        for (int i = 0; i < iters; ++i) mCtx->enqueueV3(mStream);   // 连续入队 iters 次
        DCN_CK(cudaEventRecord(t1, mStream));
        DCN_CK(cudaEventSynchronize(t1));
        float ms = 0.f;
        DCN_CK(cudaEventElapsedTime(&ms, t0, t1));    // GPU 实际耗时
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return ms / iters;                            // 单帧平均
    }

    // 内省接口。
    const std::vector<IOTensor>& io() const { return mIO; }
    std::vector<std::string> inputNames() const { return names(true); }
    std::vector<std::string> outputNames() const { return names(false); }

private:
    static std::vector<char> readFile(const std::string& p) {          // 读取整个文件到内存
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("无法打开 " + p);
        size_t n = f.tellg();
        f.seekg(0);
        std::vector<char> b(n);
        f.read(b.data(), n);
        return b;
    }
    IOTensor& at(const std::string& name) {                            // 按名查找 I/O(可写)
        for (auto& t : mIO) if (t.name == name) return t;
        throw std::runtime_error("不存在的张量: " + name);
    }
    const IOTensor& at(const std::string& name) const {                // 按名查找 I/O(只读)
        for (auto& t : mIO) if (t.name == name) return t;
        throw std::runtime_error("不存在的张量: " + name);
    }
    std::vector<std::string> names(bool wantInput) const {             // 收集输入/输出名
        std::vector<std::string> v;
        for (auto& t : mIO) if (t.isInput == wantInput) v.push_back(t.name);
        return v;
    }

    TrtLogger mLogger;                          // 日志器
    void* mPlugin{nullptr};                     // 插件 .so 句柄(dlopen)
    nvinfer1::IRuntime* mRuntime{nullptr};      // runtime
    nvinfer1::ICudaEngine* mEngine{nullptr};    // 引擎
    nvinfer1::IExecutionContext* mCtx{nullptr}; // 执行上下文
    cudaStream_t mStream{nullptr};              // 推理流
    std::vector<IOTensor> mIO;                  // 所有 I/O 张量及其显存
};

}  // namespace dcn
