#!/usr/bin/env python
"""Explicit ModelOpt Q/DQ PTQ for the dcn-trt10 backbone.

Run from learning/dcn-trt10/python after 05_backbone_to_trt.py has produced
backbone.onnx:

    python 09_ptq_modelopt_qdq.py --calib-dir calib_images --num-calib 200

This is the explicit counterpart to 06_mixed_precision.py:

* 06 uses TensorRT's legacy implicit INT8 calibrator while building.
* This script writes QuantizeLinear/DequantizeLinear nodes into ONNX first,
  then builds TensorRT from that Q/DQ graph without an INT8 calibrator.

The graph contains a custom dcn::DCNv2 node. It is excluded from ModelOpt
quantization by default and is forced to FP32 during TensorRT build so the
plugin keeps the fast im2col + cuBLAS path. The plugin must also be loaded
before calling ModelOpt because ModelOpt uses TensorRT parsing during ONNX
preprocessing.

------------------------------------------------------------------------
流水线总览(显式 QDQ = 06 隐式校准的现代版):

  输入 onnx(backbone.onnx 或 det.onnx,DCN 在 "dcn" 域)
     | quantize_qdq():
     |   (1) rewrite_dcn_domain  DCN 域 dcn->ai.onnx.contrib(ORT 才认)-> *_contrib.onnx
     |   (2) 运行期替换(monkeypatch) ort.InferenceSession,让 modelopt 内部会话挂上 dcn_ort_op 的 custom op
     |   (3) modelopt.quantize(*_contrib.onnx, 排除 DCN):ORT 跑图校准(经 PyOp 执行 DCN)-> 插 Q/DQ
     |   (4) rewrite_dcn_domain 把 QDQ 图的 DCN 域改回 "dcn" 供 TRT plugin
     v build_engine():解析 QDQ onnx -> INT8(无 calibrator)+ 强制 plugin FP32 -> .engine

  量化 detect_main 跑的网络:把 --onnx 指向 det.onnx,产物 .engine 传入 ../cpp/build/detect。
  注意:det.onnx 输出 cls/reg,不要加 --compare(compare_latency 只对 backbone 特征有意义)。
------------------------------------------------------------------------
"""

import argparse
import ctypes              # 用 CDLL 加载 DCN plugin(注册进 TensorRT)
import glob                # 扫校准图
import inspect             # 探测 modelopt.quantize 的参数(跨版本兼容)
import os                  # 路径处理
import re                  # 节点名转义(作为排除用的正则)
import time                # 延迟计时

import numpy as np
import tensorrt as trt     # 构建 INT8 引擎
import torch               # 仅 compare_latency 里建 PyTorch 参考用
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import convert_image_dtype, normalize, resize

PLUGIN = "../src/build/libdcnv2.so"        # DCN 插件库(src 下 cmake 编出)
ONNX = "backbone.onnx"                      # 默认输入图(05 导出);量化 det 时用 --onnx det.onnx
OUT_ONNX = "backbone_qdq_int8.onnx"         # 默认输出:插了 Q/DQ 的图
OUT_ENGINE = "backbone_qdq_int8.engine"     # 默认输出:最终 TRT 引擎
SIZE = 512                                  # 固定输入边长(与 05/06/08 一致)
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")  # 校准图扩展名


def list_images(image_dir, max_images):
    """递归扫描校准图目录,返回(去重排序后的)前 max_images 个路径。"""
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(image_dir, "**", ext), recursive=True)  # ** 递归子目录
    paths = sorted(set(paths))
    return paths[:max_images] if max_images > 0 else paths   # max_images<=0 表示全用


def preprocess_one(path, size):
    """单张校准图预处理:必须与推理(06/08/detect_main)逐字一致,否则校准出的 scale 偏。"""
    # Keep this aligned with 06_mixed_precision.py.
    img = read_image(path, ImageReadMode.RGB)                # 强制 3 通道 RGB(统一灰度/RGBA)
    img = resize(img, [size, size], antialias=True)          # 缩到固定尺寸
    x = convert_image_dtype(img, torch.float)                # uint8 -> [0,1]
    x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 归一化
    return x.numpy()


def preprocess_calib(calib_dir, num_images, size):
    """把校准目录里的图堆成一个 [N,3,size,size] 的 float32 数组,供 modelopt 标定使用。"""
    files = list_images(calib_dir, num_images)
    if not files:
        raise FileNotFoundError(f"no calibration images found in {calib_dir}")  # 空目录直接报错
    arr = [preprocess_one(path, size) for path in files]
    data = np.ascontiguousarray(np.stack(arr), dtype=np.float32)  # [N,3,size,size] 连续内存
    print(f"[calib] {data.shape} {data.dtype} from {len(files)} images")
    return data


def load_onnx(onnx_path):
    """读 ONNX(延迟 import onnx,避免无谓依赖)。"""
    import onnx

    return onnx.load(onnx_path)


def graph_input_name(onnx_path):
    """找出图里真正的运行期输入名(排除 weight/bias 这类 initializer),用作 calibration_data 的 key。"""
    model = load_onnx(onnx_path)
    initializer_names = {init.name for init in model.graph.initializer}  # 常量(权重)名集合
    for value in model.graph.input:
        if value.name not in initializer_names:              # 第一个非常量输入即图输入(="input")
            return value.name
    raise RuntimeError(f"cannot find a non-initializer input in {onnx_path}")


def auto_dcn_excludes(onnx_path):
    """Return regexes matching custom DCN nodes in the ONNX graph."""
    # 自动找出 DCN 节点名,作为 nodes_to_exclude 传给 modelopt -> 不给 DCN 插 Q/DQ(DCN 走 plugin FP32)。
    model = load_onnx(onnx_path)
    excludes = []
    for node in model.graph.node:
        # 三个条件任一命中即认作 DCN:域是 dcn / 算子名是 DCNv2 / 名字里含 dcn(改过域后靠 op_type 命中)
        is_dcn = node.domain == "dcn" or node.op_type == "DCNv2" or "dcn" in node.name.lower()
        if not is_dcn:
            continue
        if node.name:
            excludes.append(re.escape(node.name))            # 用转义后的节点名做精确匹配
        else:
            print("[warn] found a DCN node without node.name; pass --nodes-exclude manually if needed")
    return sorted(set(excludes))


def load_trt_plugins(plugin_path, logger=None):
    """Register DCNv2 before any TensorRT parser sees the ONNX graph."""
    # CDLL 触发插件里的 REGISTER_TENSORRT_PLUGIN -> DCNv2 Creator 入 TRT 注册表;后续解析/build 才认得它。
    # 注意:这是注册到 TensorRT,与 ORT 的 custom op(dcn_ort_op)是两套独立机制。
    ctypes.CDLL(plugin_path)
    logger = logger or trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")                  # 初始化 TRT 自带插件库
    return logger


def quantize_qdq(args):
    """显式 QDQ 量化:校准 + 插 Q/DQ,产出带 Q/DQ 的 onnx(args.out)。核心四步见函数内 (1)~(4)。"""
    try:
        from modelopt.onnx.quantization import quantize       # ModelOpt 的 ONNX flow 入口
    except ImportError as exc:
        raise RuntimeError(
            'ModelOpt is not available. Install it in the deploy env with: '
            'pip install "nvidia-modelopt[onnx]"'
        ) from exc

    import onnxruntime as ort
    import dcn_ort_op   # 手写 DCN forward + ORT custom op:让 modelopt 的 ORT 校准能执行 DCN

    load_trt_plugins(args.plugin)                            # modelopt 预处理会用到 TRT,先注册插件

    # (1) ORT(modelopt 校准用)不认自定义域 "dcn" -> 先把 DCN 节点域改成 ai.onnx.contrib 传给 modelopt。
    contrib_onnx = os.path.splitext(args.onnx)[0] + "_contrib.onnx"
    dcn_ort_op.rewrite_dcn_domain(args.onnx, contrib_onnx)            # dcn -> ai.onnx.contrib

    # (2) 运行期替换(monkeypatch):modelopt 内部会自己 new InferenceSession,逐个给它挂上我们的 DCN custom op 库。
    _orig_sess = ort.InferenceSession

    def _patched_sess(model, *a, **k):
        # 取出本次调用的 sess_options(可能在 kwargs 或第一个位置参数),没有就新建一个
        so = k.get("sess_options") or (a[0] if a and isinstance(a[0], ort.SessionOptions) else None)
        if so is None:
            so = ort.SessionOptions()
            k["sess_options"] = so
        try:
            so.register_custom_ops_library(dcn_ort_op.ort_ext_lib())  # 挂上 DCN custom op
        except Exception:  # noqa: BLE001
            pass            # 同一 SessionOptions 重复注册等情况忽略
        return _orig_sess(model, *a, **k)

    ort.InferenceSession = _patched_sess

    calib_data = preprocess_calib(args.calib_dir, args.num_calib, args.imgsz)  # 真实图校准数据
    input_name = graph_input_name(contrib_onnx)              # 校准数据要按这个输入名传入

    excludes = []
    if args.auto_exclude_dcn:
        excludes += auto_dcn_excludes(contrib_onnx)          # 按 op_type=="DCNv2" 命中(域已改不影响)
    if args.nodes_exclude:
        excludes += args.nodes_exclude                       # 也可手动补充要跳过量化的节点
    excludes = sorted(set(excludes))

    print(f"[qdq] input={input_name}")
    if excludes:
        print("[qdq] exclude nodes/patterns:")
        for item in excludes:
            print(f"      {item}")
    else:
        print("[qdq] no node exclusions")

    try:
        # (3) 调 modelopt:在 contrib 图上跑 ORT 校准(经 PyOp 执行 DCN)+ 插 Q/DQ -> 写 args.out
        quantize_kwargs = {
            "onnx_path": contrib_onnx,                               # 传入改域后的图(ORT 校准能跑 DCN)
            "quantize_mode": "int8",
            "calibration_data": {input_name: calib_data},           # {输入名: [N,3,512,512]}
            "calibration_method": args.calib_method,                # entropy / max
            "calibration_eps": args.calibration_eps,                # ORT 执行后端(cuda/cpu)
            "nodes_to_exclude": excludes or None,                   # 不量化 DCN
            "output_path": args.out,
        }
        # 下面两个参数各版本 modelopt 有无不一:有才传,避免 unexpected keyword argument
        signature = inspect.signature(quantize)
        if "high_precision_dtype" in signature.parameters:
            quantize_kwargs["high_precision_dtype"] = args.high_precision_dtype
        else:
            print("[warn] this ModelOpt version has no high_precision_dtype argument")
        if "autotune" in signature.parameters:                      # 仅在该版本支持时才传(避免 unexpected kwarg)
            quantize_kwargs["autotune"] = args.autotune

        quantize(**quantize_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "ModelOpt ONNX PTQ failed. 若仍报 DCNv2 / unregistered op,说明 ORT 没跑起 custom op:"
            "确认 dcn_ort_op.py 在 import 路径、已 pip install onnxruntime-extensions、"
            "且校准图已 rewrite_dcn_domain 到 ai.onnx.contrib。"
        ) from exc
    finally:
        ort.InferenceSession = _orig_sess                           # 还原运行期替换(monkeypatch),不污染后续 ORT 使用

    # (4) QDQ 图里 DCN 仍是 ai.onnx.contrib 域;改回 "dcn" 交给 TRT(plugin 按 op_type 匹配,见 02)。
    dcn_ort_op.rewrite_dcn_domain(args.out, args.out, to_domain="dcn")
    print(f"[qdq] done -> {args.out}(QDQ 已插入,DCN 域已改回 dcn 供 TRT plugin)")


def plugin_layer_indices(network):
    """返回网络里所有 plugin 层的下标(用于单独把它们设成 FP32)。"""
    types = [
        getattr(trt.LayerType, name)
        for name in ("PLUGIN_V3", "PLUGIN_V2", "PLUGIN")    # 兼容不同 TRT 版本的枚举名
        if hasattr(trt.LayerType, name)
    ]
    return [i for i in range(network.num_layers) if network.get_layer(i).type in types]


def parse_network(builder, logger, onnx_path):
    """解析 ONNX 成 TRT network(失败则打印解析错误并抛出)。"""
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):  # 新版 TRT 需要 explicit batch
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())                          # 解析 QDQ onnx(DCNv2 由已注册的 plugin 接住)
    if not ok:
        for i in range(parser.num_errors):
            print("   ", parser.get_error(i))
        raise RuntimeError(f"parse failed: {onnx_path}")
    return network


def build_engine(args):
    """从 QDQ onnx 构建 INT8 引擎:Q/DQ 自带 scale(无 calibrator),plugin 层强制 FP32。"""
    logger = load_trt_plugins(args.plugin)                   # 再次确保插件已注册

    builder = trt.Builder(logger)
    network = parse_network(builder, logger, args.out)       # 解析的是 quantize_qdq 产出的 QDQ 图
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)  # workspace 上限(GiB)

    # Q/DQ carries the scales. No calibrator is attached here.
    config.set_flag(trt.BuilderFlag.INT8)                    # 允许 INT8 内核;scale 来自图里的 Q/DQ
    if args.trt_fp16:
        config.set_flag(trt.BuilderFlag.FP16)                # 非量化层可用 FP16 tactic
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)  # 让我们手动指定的层精度被优先尊重

    for idx in plugin_layer_indices(network):                # 把每个 DCN plugin 层固定为 FP32
        layer = network.get_layer(idx)
        try:
            layer.precision = trt.DataType.FLOAT             # FP32 才命中 im2col+cuBLAS 快速路径
            for out_idx in range(layer.num_outputs):
                layer.set_output_type(out_idx, trt.DataType.FLOAT)
            print(f"[trt] force plugin FP32: layer {idx} {layer.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] cannot force plugin layer {idx} to FP32: {exc}")

    print(f"[trt] building Q/DQ engine from {args.out} ...")
    plan = builder.build_serialized_network(network, config) # 构建并序列化引擎
    if plan is None:
        raise RuntimeError("TensorRT build failed")
    plan = bytes(plan)
    if args.engine:
        with open(args.engine, "wb") as f:
            f.write(plan)                                    # 落盘 .engine
        print(f"[trt] wrote {args.engine} ({len(plan) / 1e6:.1f} MB)")
    return plan, network.get_input(0).name, [network.get_output(i).name for i in range(network.num_outputs)]


def compare_latency(engine_bytes, input_name, output_names, runs):
    """对比引擎输出与 PyTorch backbone 特征 + 测 P50 延迟。
    注意:仅对 backbone(输出 FPN 特征)有意义;det.onnx 输出 cls/reg,形状对不上会失败 -> det 别加 --compare。"""
    from detector import build_dcn_detector
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    model, _ = build_dcn_detector()
    backbone = model.backbone.eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = backbone(x)                                    # PyTorch 参考特征(多尺度 dict)
    keys = list(ref.keys())

    x_np = np.ascontiguousarray(x.numpy(), np.float32)
    with TrtRunner(EngineFromBytes(engine_bytes)) as runner:
        res = runner.infer({input_name: x_np})               # 取一次输出比精度
        for _ in range(10):
            runner.infer({input_name: x_np})                 # warmup
        times = []
        for _ in range(runs):                                # 测 runs 次延迟
            t0 = time.perf_counter()
            runner.infer({input_name: x_np})
            times.append((time.perf_counter() - t0) * 1e3)

    worst = 0.0
    for i, key in enumerate(keys):                           # 各尺度特征 vs PyTorch 的最差误差
        got = np.array(res[output_names[i]], np.float32).reshape(ref[key].shape)
        err = float(np.abs(got - ref[key].numpy()).max())
        worst = max(worst, err)
    times.sort()
    print(f"[compare] worst feature max|err|={worst:.3e}, P50={times[len(times) // 2]:.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=ONNX)                          # 输入图;量化 det 时传 det.onnx
    parser.add_argument("--plugin", default=PLUGIN)                      # DCN 插件 .so
    parser.add_argument("--calib-dir", default="calib_images")           # 校准图目录
    parser.add_argument("--num-calib", type=int, default=200)            # 校准图张数
    parser.add_argument("--imgsz", type=int, default=SIZE)               # 输入边长
    parser.add_argument("--calib-method", choices=["entropy", "max"], default="entropy")  # 标定算法
    parser.add_argument(
        "--high-precision-dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="ModelOpt dtype for unquantized/high-precision ONNX tensors. "
             "fp32 avoids ModelOpt fp16 autocast issues with custom DCN shape info.",
    )
    parser.add_argument("--calibration-eps", nargs="+", default=["cuda:0", "cpu"])  # ORT 校准执行后端
    parser.add_argument("--nodes-exclude", nargs="*", default=None)      # 手动追加不量化的节点
    parser.add_argument("--no-auto-exclude-dcn", dest="auto_exclude_dcn", action="store_false")  # 关掉自动排除 DCN
    parser.set_defaults(auto_exclude_dcn=True)
    parser.add_argument("--autotune", action="store_true")               # modelopt autotune(若该版本支持)
    parser.add_argument("--out", default=OUT_ONNX)                       # 输出 QDQ onnx
    parser.add_argument("--engine", default=OUT_ENGINE)                  # 输出 .engine
    parser.add_argument("--workspace", type=int, default=2, help="TensorRT workspace in GiB")
    parser.add_argument(
        "--trt-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TensorRT to use FP16 tactics for non-QDQ layers during engine build.",
    )
    parser.add_argument("--skip-quantize", action="store_true")          # 跳过量化(用已有 QDQ onnx 直接 build)
    parser.add_argument("--skip-build", action="store_true")             # 只量化、不 build 引擎
    parser.add_argument("--compare", action="store_true")                # 跑精度+延迟对比(仅 backbone 用)
    parser.add_argument("--runs", type=int, default=100)                 # 延迟测量次数
    args = parser.parse_args()

    if not args.skip_quantize:
        quantize_qdq(args)                                   # 校准 + 插 Q/DQ -> args.out
    if args.skip_build:
        return

    engine, input_name, output_names = build_engine(args)    # QDQ onnx -> INT8 .engine
    if args.compare:
        compare_latency(engine, input_name, output_names, args.runs)


if __name__ == "__main__":
    main()
