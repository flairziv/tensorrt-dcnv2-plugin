#!/usr/bin/env python
"""含 DCN 的 backbone 导出 ONNX -> 构建 TRT FP32 引擎 -> 与 PyTorch backbone 特征逐尺度比对。

验证 DCN plugin 在完整 ResNet50 + FPN(数百层)中正常工作,而非仅孤立节点:
  导出 model.backbone(BackboneWithFPN,输出多尺度 FPN 特征,dynamo=False 以保留 dcn::DCNv2 节点)
  -> 构建 TRT FP32 引擎(DCN 走 plugin,其余 ResNet/FPN 走原生)-> 同一输入下逐特征图比对数值。

运行(WSL,dl env):  python 05_backbone_to_trt.py
"""
import ctypes              # 加载插件 .so

import numpy as np         # 数值比对
import torch               # PyTorch 参考与导出
import tensorrt as trt     # 构建引擎

from detector import build_dcn_detector   # 构建含 DCN 的检测器

PLUGIN = "../src/build/libdcnv2.so"   # 插件库
SIZE = 512   # 固定输入边长(32 的倍数,适配 FPN 各尺度)


def main():
    model, _ = build_dcn_detector()         # 构建检测器(此处仅用其 backbone)
    backbone = model.backbone.eval()         # 取 backbone(ResNet50 + FPN,含一层 DCN)
    x = torch.randn(1, 3, SIZE, SIZE)        # 示例输入 [1,3,512,512]

    with torch.no_grad():
        ref = backbone(x)                       # OrderedDict:多尺度 FPN 特征
    keys = list(ref.keys())
    out_names = [f"feat{i}" for i in range(len(keys))]   # ONNX 输出命名 feat0/1/...
    print(f"[export] backbone 输出: {[(k, tuple(ref[k].shape)) for k in keys]}")

    torch.onnx.export(backbone, x, "backbone.onnx", input_names=["input"],
                      output_names=out_names, opset_version=17,
                      custom_opsets={"dcn": 1}, dynamo=False)   # dynamo=False 才能将 DCN 导出为 dcn::DCNv2 节点
    import onnx
    doms = {n.domain for n in onnx.load("backbone.onnx").graph.node}
    print(f"[export] backbone.onnx 已生成;含 dcn 自定义节点: {'dcn' in doms}(应为 True)")

    # 构建 TRT FP32 引擎(DCN 走 plugin)。
    ctypes.CDLL(PLUGIN)                      # 加载插件并注册
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) \
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") else 0
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open("backbone.onnx", "rb").read()):   # 解析(数百层 + 1 个 DCN plugin 节点)
        for i in range(parser.num_errors):
            print("   ", parser.get_error(i).desc())
        raise SystemExit("parse 失败")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)   # workspace 2GiB(大网络需更多)
    print("[trt] 正在构建 FP32 引擎(ResNet50 + FPN,约 30-60s)...")
    plan = bytes(builder.build_serialized_network(net, cfg))
    print(f"[trt] FP32 引擎构建成功: {len(plan) / 1e6:.1f} MB")

    # 逐特征图推理比对。
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    in_name = net.get_input(0).name
    with TrtRunner(EngineFromBytes(plan)) as r:
        res = r.infer({in_name: np.ascontiguousarray(x.numpy(), np.float32)})   # 同一输入运行 TRT
    print("[compare] TRT backbone vs PyTorch backbone:")
    worst = 0.0
    for i, k in enumerate(keys):
        g = ref[k].numpy()                   # PyTorch 参考特征
        t = np.array(res[out_names[i]], np.float32).reshape(g.shape)   # TRT 对应输出
        e = float(np.abs(t - g).max())       # 最大绝对误差
        worst = max(worst, e)
        print(f"   {k}: shape{tuple(g.shape)}  max|err|={e:.3e}")
    print(f"\n最差 max|err|={worst:.3e}: "
          f"{'DCN plugin 在 ResNet50 + FPN 中工作正常(特征对齐)' if worst < 1e-2 else '偏差较大,检查 DCN 节点 / plugin'}")
    print("  下一步:同一图构建混合精度引擎(backbone INT8 + DCN FP32)并对比延迟。")


if __name__ == "__main__":
    main()
