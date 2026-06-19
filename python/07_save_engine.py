#!/usr/bin/env python
"""构建 FP16 backbone 引擎并保存到磁盘,供 C++ 部署使用。需先运行 05 生成 backbone.onnx。

运行:  python 07_save_engine.py
"""
import ctypes              # 加载插件 .so

import tensorrt as trt     # 构建与序列化引擎

PLUGIN = "../src/build/libdcnv2.so"     # 插件库
ONNX = "backbone.onnx"                   # 05 导出的 backbone 图
OUT = "backbone_fp16.engine"             # 输出引擎文件(供 cpp/dcn_infer 加载)


def main():
    ctypes.CDLL(PLUGIN)                  # 加载插件并注册(构建与反序列化均需)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) \
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") else 0   # 显式 batch(兼容)
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(ONNX, "rb").read()):
        raise SystemExit("parse fail")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)   # 2GiB workspace
    cfg.set_flag(trt.BuilderFlag.FP16)              # 启用 FP16(backbone 走 FP16 提速并省显存)
    # 将 DCN plugin 层固定为 FP32,以命中其 im2col + cuBLAS 快速路径;backbone 仍走 FP16。
    cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)   # 使下面手动指定的层精度被尊重
    ptypes = [getattr(trt.LayerType, n) for n in ("PLUGIN_V3", "PLUGIN_V2", "PLUGIN")   # 兼容不同版本的 plugin 层枚举
              if hasattr(trt.LayerType, n)]
    for i in range(net.num_layers):
        if net.get_layer(i).type in ptypes:         # 命中 plugin 层(即 DCN)
            try:
                net.get_layer(i).precision = trt.DataType.FLOAT   # 将 DCN 固定为 FP32,enqueue 走最快的 im2col + cuBLAS 路径
            except Exception:  # noqa: BLE001
                pass
    plan = bytes(builder.build_serialized_network(net, cfg))   # 构建并序列化引擎字节
    with open(OUT, "wb") as f:
        f.write(plan)
    print(f"已保存 {OUT}({len(plan) / 1e6:.1f} MB),供 C++ 部署使用")
    print("   注:引擎绑定本机 GPU(sm89)与 TRT 10.16,更换显卡或版本需重新构建")


if __name__ == "__main__":
    main()
