#!/usr/bin/env python
"""DCNv2 plugin 的 FP16 路径验证与延迟基准。

两部分:
  A) 精度:小图(1x16x32x32)分别以 FP32 / FP16 运行,与 torchvision oracle(dcn_io.npz 的 y)比对;
  B) 延迟:大图(1x64x128x128,随机权重)FP32 与 FP16 的 P50/P90 对比。

运行(WSL,dl env;需先编译含 FP16 路径的 libdcnv2.so):
  python 03_benchmark.py
"""
import ctypes              # 加载插件 .so
import os                  # 路径相关(预留)
import time                # 计时(perf_counter)

import numpy as np         # 数值与比对
import onnx                # 构造图
import tensorrt as trt     # 构建引擎
from onnx import TensorProto, helper, numpy_helper   # 构造 onnx 节点/图/常量

PLUGIN = "../src/build/libdcnv2.so"   # 插件库路径


def make_dcn_onnx(path, N, Cin, Cout, H, W, K=3, stride=1, pad=1, dil=1, seed=0):
    """构造指定尺寸的 DCNv2 图(随机 weight/bias 常量),返回 (K, Ho, Wo)。"""
    rng = np.random.default_rng(seed)              # 固定种子,可复现
    Ho = (H + 2 * pad - dil * (K - 1) - 1) // stride + 1   # 输出高(与 plugin getOutputShapes 同公式)
    Wo = (W + 2 * pad - dil * (K - 1) - 1) // stride + 1   # 输出宽
    w = (rng.standard_normal((Cout, Cin, K, K)) * 0.1).astype(np.float32)   # 随机权重常量
    b = np.zeros((Cout,), np.float32)              # 零偏置常量
    node = helper.make_node("DCNv2", ["input", "offset", "mask", "weight", "bias"], ["output"],   # DCNv2 节点(同 01 契约)
                            domain="dcn", stride=stride, padding=pad, dilation=dil,
                            kernel=K, deformable_groups=1)
    g = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, Cin, H, W]),
         helper.make_tensor_value_info("offset", TensorProto.FLOAT, [N, 2 * K * K, Ho, Wo]),
         helper.make_tensor_value_info("mask", TensorProto.FLOAT, [N, K * K, Ho, Wo])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, Cout, Ho, Wo])],
        [numpy_helper.from_array(w, "weight"), numpy_helper.from_array(b, "bias")])         # weight/bias 常量
    onnx.save(helper.make_model(
        g, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("dcn", 1)]), path)
    return K, Ho, Wo


def build(logger, onnx_path, fp16, force_half=False):
    """解析 onnx 并构建引擎;fp16 开启 FP16 标志,force_half 强制 I/O 为 FP16。返回 (引擎字节, 输入名列表)。"""
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) \
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") else 0   # 显式 batch 标志(兼容)
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        raise SystemExit("parse fail: " + parser.get_error(0).desc())
    if force_half:   # 强制 I/O 为 FP16,使 plugin 只能选 all-FP16 组合,从而真正执行 half kernel
        for i in range(net.num_inputs):
            net.get_input(i).dtype = trt.DataType.HALF
        for i in range(net.num_outputs):
            net.get_output(i).dtype = trt.DataType.HALF
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)   # workspace 1GiB
    if fp16 or force_half:
        cfg.set_flag(trt.BuilderFlag.FP16)                # 允许使用 FP16
    return bytes(builder.build_serialized_network(net, cfg)), \
        [net.get_input(i).name for i in range(net.num_inputs)]


def main():
    ctypes.CDLL(PLUGIN)                          # 加载插件并注册
    logger = trt.Logger(trt.Logger.ERROR)        # 仅打印 ERROR,减少基准噪声
    trt.init_libnvinfer_plugins(logger, "")
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    # A) 精度:小图 vs oracle。
    d = np.load("dcn_io.npz")
    print("== A) 精度(小图 1x16x32x32,对 torchvision oracle)==")
    for fp16 in (False, True):
        plan, names = build(logger, "dcn.onnx", fp16)
        feeds = {n: np.ascontiguousarray({"input": d["x"], "offset": d["offset"],
                                          "mask": d["mask"]}[n], np.float32) for n in names}
        with TrtRunner(EngineFromBytes(plan)) as r:
            y = np.array(list(r.infer(feeds).values())[0], np.float32).reshape(d["y"].shape)
        print(f"   {'FP16' if fp16 else 'FP32'}: max|err vs oracle| = {np.abs(y - d['y']).max():.3e}")
    # 强制 FP16 I/O:验证 half kernel 确实被执行(误差应升至约 1e-2)。
    try:
        plan, names = build(logger, "dcn.onnx", True, force_half=True)
        src = {"input": d["x"], "offset": d["offset"], "mask": d["mask"]}
        feeds = {n: np.ascontiguousarray(src[n], np.float16) for n in names}
        with TrtRunner(EngineFromBytes(plan)) as r:
            y = np.array(list(r.infer(feeds).values())[0], np.float32).reshape(d["y"].shape)
        print(f"   FP16(强制 I/O): max|err| = {np.abs(y - d['y']).max():.3e}"
              f"   (升至约 1e-2 表明 half kernel 确已执行)")
    except Exception as e:  # noqa: BLE001
        print(f"   FP16(强制 I/O): 跳过({type(e).__name__}: {e})")   # 部分版本不允许设置 I/O dtype

    # B) 延迟:大图 FP32 vs FP16。
    N, Cin, Cout, H, W = 1, 64, 64, 128, 128     # 较大尺寸以体现延迟差异
    K, Ho, Wo = make_dcn_onnx("dcn_big.onnx", N, Cin, Cout, H, W)
    rng = np.random.default_rng(1)
    big = {"input": rng.standard_normal((N, Cin, H, W)).astype(np.float32),
           "offset": rng.standard_normal((N, 2 * K * K, Ho, Wo)).astype(np.float32),
           "mask": rng.random((N, K * K, Ho, Wo)).astype(np.float32)}             # mask in [0,1)
    print(f"\n== B) 延迟(大图 {N}x{Cin}x{H}x{W},含 H2D/D2H 与同步,P50/P90 ms)==")
    for fp16 in (False, True):
        plan, names = build(logger, "dcn_big.onnx", fp16)
        feeds = {n: np.ascontiguousarray(big[n], np.float32) for n in names}
        with TrtRunner(EngineFromBytes(plan)) as r:
            for _ in range(20):
                r.infer(feeds)                       # warmup
            ts = []
            for _ in range(300):
                t0 = time.perf_counter()
                r.infer(feeds)
                ts.append((time.perf_counter() - t0) * 1e3)   # 毫秒
        ts.sort()
        print(f"   {'FP16' if fp16 else 'FP32'}: P50={ts[len(ts) // 2]:.3f}  P90={ts[int(len(ts) * 0.9)]:.3f}")

    print("\n说明:FP16 精度约 1e-2 属正常(half I/O);若与 FP32 同为约 1e-7,表示 TRT 对该小型 plugin 仍选用 FP32。")
    print("     本 plugin 未做 shared memory / 向量化优化,FP16 提速有限属预期。")


if __name__ == "__main__":
    main()
