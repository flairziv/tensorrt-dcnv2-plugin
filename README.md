# dcn-trt10

> TensorRT 10(IPluginV3)可变形卷积 DCNv2 插件:量化感知 + im2col/cuBLAS 加速 + 真实检测器端到端部署(纯 C++)。

将通常基于已弃用接口(IPluginV2)且仅支持浮点的 DCN 插件,实现为现代 IPluginV3 + FP32/FP16/INT8
+ im2col-cuBLAS 加速 + 真实检测器混合精度部署的工程化版本。

---

## 解决的问题

- TensorRT 无 DCN 算子,含 DCN 的 ONNX 解析时报 `checkFallbackPluginImporter: Plugin not found`。
- 公开的 DCN-TRT 插件多基于 TRT 8 及以前的 `IPluginV2DynamicExt`(已弃用,TRT10 无法编译),且仅支持浮点。
- 本项目以 TRT 10 `IPluginV3` 实现 DCNv2,支持 FP32 / FP16 / INT8,采用 im2col + cuBLAS GEMM 加速,
  并在真实 RetinaNet 检测器上端到端部署,以纯 C++ runtime 运行。

## 主要特性

1. 现代 `IPluginV3` 接口(TRT 10),而非已弃用的 `IPluginV2DynamicExt`。
2. 量化感知:INT8 DCN kernel(x/weight 对称量化,平均相对误差 1.27%)+ 混合精度配方
   (backbone 低精度 + DCN 高精度)。DCN 不易量化,在 INT8 管线中的精度安置是实际工程问题。
3. im2col + cuBLAS GEMM 加速:将卷积转化为矩阵乘交由 tensor core,延迟 40 ms 降至 2.6 ms(约 15 倍)。
4. 真实检测器端到端 + 纯 C++ 部署:RetinaNet 插入一层 DCN,导出 ONNX,构建 TRT 引擎,纯 C++ runtime 出框。
5. 三种后端自定义算子对照(TensorRT / OpenVINO / RKNN,见文末)。

## 性能(RTX 4060 Laptop;backbone = ResNet50+FPN+DCN @ 512²;纯 C++ cudaEvent,不含 H2D/D2H)

| DCN kernel 版本 | 延迟 | 优化点 |
|---|---|---|
| naive(每输出一线程) | 40.2 ms | 每个输出通道重复采样,Cout=256 倍冗余 |
| + hoist offset/mask | 29.2 ms | offset/mask 不依赖输入通道,省 Cin 倍冗余读取(访存约 -37%) |
| + im2col + cuBLAS GEMM | 2.59 ms | 每个采样仅计算一次(消除 Cout 冗余)+ 卷积交由 cuBLAS tensor core |

> Python(polygraphy)端三种精度均约 42 ms,其延迟被解释器与 H2D/D2H 开销主导;纯 C++ cudaEvent 才反映真实算力延迟。

## 精度

| 检查项 | 结果 |
|---|---|
| DCN kernel vs torchvision oracle(FP32) | 3.6e-7 |
| im2col + cuBLAS GEMM vs oracle | 8.3e-7 |
| TRT 引擎 vs PyTorch backbone 特征(FP32) | 最差 1.4e-3 |
| INT8 DCN(x/weight 对称量化) | 平均相对 1.27% |

## 流水线

```
PyTorch RetinaNet(FPN 中插入一层 identity-init DCN)
   │  04_detect_image.py:真实图检测(验证插入 DCN 后仍正常检测)
   ▼
导出 ONNX(dcn::DCNv2 自定义节点,torch.onnx.export dynamo=False)
   ▼
TensorRT 引擎(DCN 走本插件;backbone FP16 或 INT8 校准 = 混合精度)
   │  05/06:特征对齐 + 混合精度基准
   │  08:导出含检测头的 det.engine + 预生成 anchors.bin + numpy 后处理参考
   ▼
纯 C++ runtime:dlopen 插件 -> deserialize -> enqueueV3
   - dcn_infer:backbone 引擎延迟基准(约 2.6 ms/帧)
   - detect:含检测头引擎 + anchor 解码 + class-aware NMS,端到端出框
```

DCN 插件内部:`enqueue` 按精度分派 —— FP32 走 im2col + cuBLAS GEMM(快速路径);FP16/INT8 走逐元素 kernel(见"工程边界")。

## 目录

```
src/   dcn_kernel.{h,cu}  DCN kernel(naive FP32/FP16/INT8 + im2col + bias)
       dcn_plugin.cpp     IPluginV3 + Creator + REGISTER_TENSORRT_PLUGIN
       test_dcn*.cu       独立数值验证(FP32 / im2col-GEMM / INT8)
       CMakeLists.txt     -> libdcnv2.so
python/ 01 oracle + 导出自定义节点 ONNX / 02 plugin 端到端对齐
        03 FP32 / FP16 基准 / detector.py(RetinaNet + DCN)/ 04 真实检测
        05 backbone -> TRT 特征对齐 / 06 混合精度 / 07 保存 backbone 引擎
        08 导出含检测头引擎 + anchors + 参考 / probe_export.py 导出路径探查
cpp/   trt_engine.h       可复用 TRT 引擎封装(RAII)
       dcn_detector.h     RetinaNet 后处理(anchor 解码 + class-aware NMS)
       dcn_infer.cpp      backbone 引擎部署 + cudaEvent 延迟基准
       detect_main.cpp    纯 C++ 端到端检测(可选 OpenCV 读图/画框)
```

## 快速复现(WSL + CUDA 12.6 + TensorRT 10.16;conda 环境含 torch/torchvision/onnx/polygraphy)

```bash
# 1) 编译插件
cd src
TRT_INC=$(realpath ../../trt_quantize/TensorRT-10.16/include)
NVINFER=$(python -c "import tensorrt_libs,glob,os;print(glob.glob(os.path.join(os.path.dirname(tensorrt_libs.__file__),'libnvinfer.so*'))[0])")
cmake -B build -DTRT_INCLUDE_DIR=$TRT_INC -DTRT_NVINFER=$NVINFER && cmake --build build -j

# 2) 算子正确性(独立单元测试)
cd ../python
python 01_oracle_and_export.py   # 生成 oracle .bin
nvcc -O2 -std=c++17 -arch=sm_89 ../src/test_dcn_gemm.cu ../src/dcn_kernel.cu -I../src -lcublas -o t && ./t

# 3) 真实检测示例(PyTorch)
python 04_detect_image.py image.jpg          # -> det_out.jpg

# 4) backbone -> TRT + 混合精度(06 需在 calib_images/ 放数十张真实图)
python 05_backbone_to_trt.py
python 06_mixed_precision.py

# 5) 保存 backbone 引擎 + 纯 C++ 延迟基准
python 07_save_engine.py
cd ../cpp && cmake -B build -DTRT_INCLUDE_DIR=$TRT_INC -DTRT_NVINFER=$NVINFER && cmake --build build -j
cd ../python
export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))"):$LD_LIBRARY_PATH
../cpp/build/dcn_infer backbone_fp16.engine ../src/build/libdcnv2.so

# 6) 含检测头的纯 C++ 端到端检测
python 08_export_det_engine.py image.jpg     # 生成 det.engine / anchors.bin / det_*.txt
../cpp/build/detect det.engine ../src/build/libdcnv2.so .            # 与 Python 参考对齐
../cpp/build/detect det.engine ../src/build/libdcnv2.so . image.jpg  # 直接读图并画框(需 OpenCV)
```

## 工程边界

- DCN 不易量化:TRT 量化器不会自动量化 plugin,仅按 `supportsFormatCombination` 声明的精度运行。
  实务推荐混合精度(backbone 低精度 + DCN 高精度),而非强制 DCN 走 INT8(收益有限且精度风险较高)。
- 快速路径当前为 FP32(im2col + `cublasSgemm`);FP16/INT8 仍走逐元素 kernel。FP16 GEMM
  (`cublasGemmEx` 走 tensor core)为后续优化方向。
- 拆分部署:检测头与 NMS 置于 PyTorch / C++,重计算(backbone + DCN)上 TRT,为常见边缘部署形态。
- DCN 采用 identity 初始化(offset 约 0,mask 约 1)插入预训练检测器,无需训练即保持检测效果,聚焦部署而非训练精度。
- 引擎绑定 GPU 架构与 TRT 版本,更换显卡或版本需重新构建。
- INT8 混合精度采用真实图熵校准(06_mixed_precision.py 的 EntropyCalibrator,需在 calib_images/ 放置真实图)。

## 三种后端自定义算子对照

| 后端 | 自定义算子机制 | DCN 落地 |
|---|---|---|
| TensorRT | `IPluginV3`(C++ + CUDA kernel) | 本项目;GPU 上可接入任意 kernel |
| OpenVINO | Custom Operation(C++/OpenCL) | 可行,但需实现 shape 推断与 evaluator,较繁琐 |
| RKNN | 仅 CPU fallback | NPU 不支持 DCN,只能标记该层回退 CPU,无法获得 NPU 加速 |
