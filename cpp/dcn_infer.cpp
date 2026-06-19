// dcn_infer.cpp —— 边缘部署形态的纯 C++ TensorRT 推理示例:使用 TrtEngine 运行引擎并做延迟基准。
//
//   可复用逻辑封装于 trt_engine.h,本文件仅演示用法(加载 -> 写入输入 -> 推理 -> 计时)。
//   引擎为 backbone(ResNet50 + FPN + DCN),输出 FPN 特征图;检测头与 NMS 在 PyTorch(拆分部署)。
//   以 cudaEvent 测纯 GPU 延迟(不含 H2D/D2H,无 Python 开销),反映边缘端真实性能。
//
// 编译见同目录 CMakeLists.txt;运行(在 python/ 目录,引擎位于此处):
//   ../cpp/build/dcn_infer backbone_fp16.engine ../src/build/libdcnv2.so

#include "trt_engine.h"   // 引擎封装类(RAII;加载/显存管理/推理/计时)

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    const char* engPath = argc > 1 ? argv[1] : "backbone_fp16.engine";        // 第 1 参数:引擎路径
    const char* pluginPath = argc > 2 ? argv[2] : "../src/build/libdcnv2.so"; // 第 2 参数:插件 .so

    try {
        dcn::TrtEngine engine(engPath, pluginPath);   // 加载插件 + 反序列化引擎 + 发现并分配所有 I/O
        printf("[1] 引擎与插件已加载\n");

        // 遍历 I/O:打印形状,并为每个输入写入确定性伪数据(仅用于链路验证与延迟测试)。
        for (const auto& t : engine.io()) {
            printf("    %s %-8s [", t.isInput ? "IN " : "OUT", t.name.c_str());
            for (int k = 0; k < t.dims.nbDims; ++k)
                printf("%d%s", (int)t.dims.d[k], k + 1 < t.dims.nbDims ? "," : "");
            printf("]\n");
            if (t.isInput) {                          // 输入张量:写入伪数据
                std::vector<float> h(t.vol);
                for (size_t z = 0; z < t.vol; ++z) h[z] = (float)(z % 255) / 255.f - 0.5f;
                engine.setInput(t.name, h.data());    // host -> device
            }
        }

        engine.infer();                               // 执行一次以验证链路
        printf("[2] 推理完成(输出为 FPN 特征图;最终框由 PyTorch 端的 head + NMS 产生)\n");
        // 如需取输出做后续处理:auto feat = engine.getOutput(engine.outputNames()[0]);

        float ms = engine.benchmark(200);             // 纯 GPU 延迟(200 次平均)
        printf("[3] 纯 C++ 推理延迟:%.3f ms/帧(纯 GPU,不含 H2D/D2H)\n", ms);
    } catch (const std::exception& e) {               // TrtEngine 出错抛异常,统一处理
        fprintf(stderr, "错误: %s\n", e.what());
        return 1;
    }
    return 0;
}
