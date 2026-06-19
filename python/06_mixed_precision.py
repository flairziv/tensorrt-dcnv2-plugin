#!/usr/bin/env python
"""基于同一 backbone.onnx 构建 FP32 / FP16 / 混合精度 三个引擎,对比延迟与特征精度。

混合精度方案:backbone 走 INT8(真实图校准)+ DCN plugin 保持 FP32(命中 im2col + cuBLAS 快速路径)。
  DCN 不易量化、精度敏感且算量小,故保留高精度;backbone 为计算主体,走 INT8 以提速并省显存。
  DCN 必须为 FP32 才命中快速路径;强制 FP16 会退回较慢的朴素实现。

需先运行 05 生成 backbone.onnx,并在 CALIB_DIR 放置数十张真实图。运行:  python 06_mixed_precision.py
"""
import ctypes              # 加载插件 .so
import glob                 # 扫描校准图
import os                   # 路径与缓存判断
import time                # 计时

import numpy as np         # 数值比对
import torch               # PyTorch 参考与校准张量
import tensorrt as trt     # 构建引擎与 INT8 校准器基类
from torchvision.io import read_image, ImageReadMode                       # 读校准图(强制 RGB)
from torchvision.transforms.functional import resize, convert_image_dtype, normalize  # 校准图预处理

from detector import build_dcn_detector   # 构建含 DCN 的检测器

PLUGIN = "../src/build/libdcnv2.so"   # 插件库
ONNX = "backbone.onnx"                 # 05 导出的 backbone 图
SIZE = 512                             # 输入边长
CALIB_DIR = "calib_images"             # 校准图目录(jpg/png),mixed 模式使用


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 熵校准器:使用真实图进行校准并将结果落盘缓存。

    注意:int8_calibrator 在 TRT10 已 deprecated,现代做法为 modelopt 显式 QDQ;此处保留以演示隐式校准。
    """

    def __init__(self, img_dir, cache="calib.cache", batch=1, max_imgs=200):
        super().__init__()
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")  # 兼容多扩展名与大小写
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(img_dir, e))
        self.paths = sorted(set(paths))[:max_imgs]
        assert self.paths, (                                      # 空目录直接报错,避免在 0 张数据上校准出错误 scale
            f"校准目录 '{img_dir}' 为空,请放入约 16-200 张真实自然图(jpg/png)。")
        self.batch, self.cache_path, self.idx = batch, cache, 0
        self.dbuf = torch.empty(batch, 3, SIZE, SIZE, device="cuda")  # 预分配显存,整个校准过程复用
        print(f"[calib] 使用 {len(self.paths)} 张真实图校准(目录 {img_dir})")

    def get_batch_size(self):
        return self.batch

    def _preprocess(self, path):
        # 须与推理预处理一致:引擎为去除 transform 的 backbone,需输入已 ImageNet 归一化的张量(同 08/detect)。
        img = read_image(path, ImageReadMode.RGB)                 # 强制 3 通道 RGB(统一灰度/RGBA,避免通道数错误)
        img = resize(img, [SIZE, SIZE], antialias=True)           # 缩放至固定尺寸
        x = convert_image_dtype(img, torch.float)                 # uint8 -> [0,1]
        return normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 减均值除标准差,与推理对齐

    def get_batch(self, names):                                   # TRT 反复调用以获取下一批数据的设备指针
        if self.idx + self.batch > len(self.paths):
            return None                                           # 数据用尽,返回 None 表示校准结束
        for b in range(self.batch):
            self.dbuf[b] = self._preprocess(self.paths[self.idx + b]).cuda()  # 预处理后拷入复用显存
        self.idx += self.batch
        return [int(self.dbuf.data_ptr())]                        # 按 names 顺序返回 device 指针(此处单输入)

    def read_calibration_cache(self):
        # 存在缓存则使用,跳过重复校准。更换模型或预处理时须删除该 .cache,否则读到旧 scale。
        return open(self.cache_path, "rb").read() if os.path.exists(self.cache_path) else None

    def write_calibration_cache(self, cache):
        open(self.cache_path, "wb").write(cache)                  # 校准完成后将 scale 表落盘


def _plugin_layers(net):
    """返回网络中所有 plugin 层的下标(用于单独设置其精度)。"""
    types = [getattr(trt.LayerType, n) for n in ("PLUGIN_V3", "PLUGIN_V2", "PLUGIN")   # 兼容不同 TRT 版本的枚举名
             if hasattr(trt.LayerType, n)]
    return [i for i in range(net.num_layers) if net.get_layer(i).type in types]


def build(logger, mode, calib=None):
    """按 mode(fp32 / fp16 / mixed)构建引擎。返回 (引擎字节, 输入名, 输出名列表)。"""
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) \
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") else 0
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(ONNX, "rb").read()):   # 解析 backbone.onnx
        raise SystemExit("parse fail")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)   # 2GiB workspace
    if mode in ("fp16", "mixed"):
        cfg.set_flag(trt.BuilderFlag.FP16)          # fp16 / mixed 均允许 FP16
    if mode == "mixed":
        cfg.set_flag(trt.BuilderFlag.INT8)          # 额外允许 INT8
        cfg.int8_calibrator = calib                 # 挂载 INT8 校准器(决定各层 scale)
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)   # 使手动指定的层精度被优先尊重
        for li in _plugin_layers(net):                 # 将 DCN plugin 层设为 FP32 以命中 im2col + cuBLAS 快速路径
            try:
                net.get_layer(li).precision = trt.DataType.FLOAT  # FP32 才走快速路径;设为 HALF 会退回较慢的朴素实现
            except Exception:  # noqa: BLE001
                pass
    plan = bytes(builder.build_serialized_network(net, cfg))
    return plan, net.get_input(0).name, [net.get_output(i).name for i in range(net.num_outputs)]


def main():
    model, _ = build_dcn_detector()
    backbone = model.backbone.eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = backbone(x)                    # PyTorch 参考特征(多尺度)
    keys = list(ref.keys())
    xnp = np.ascontiguousarray(x.numpy(), np.float32)

    ctypes.CDLL(PLUGIN)                      # 加载插件
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    calib = EntropyCalibrator(CALIB_DIR)     # 真实图校准器(mixed 模式使用)

    print(f"{'引擎':<8}{'最差特征 err':>16}{'P50 ms':>10}")
    for mode in ("fp32", "fp16", "mixed"):
        plan, inn, outs = build(logger, mode, calib if mode == "mixed" else None)   # 仅 mixed 传校准器
        with TrtRunner(EngineFromBytes(plan)) as r:
            res = r.infer({inn: xnp})        # 先运行一次取输出(用于精度比对)
            for _ in range(10):
                r.infer({inn: xnp})                    # warmup
            ts = []
            for _ in range(100):             # 测量 100 次延迟
                t0 = time.perf_counter()
                r.infer({inn: xnp})
                ts.append((time.perf_counter() - t0) * 1e3)
        worst = max(float(np.abs(np.array(res[outs[i]], np.float32).reshape(ref[keys[i]].shape)   # 各尺度特征 vs PyTorch 的最差误差
                                 - ref[keys[i]].numpy()).max()) for i in range(len(keys)))
        ts.sort()
        print(f"{mode:<8}{worst:>16.3e}{ts[len(ts) // 2]:>10.3f}")

    print("\n混合精度 = backbone INT8(真实图校准)+ DCN FP32(im2col + cuBLAS 快速路径)。")
    print("  - DCN 保持 FP32 为有意设计:不易量化、精度敏感且算量小;快速路径仅支持 FP32(强制 FP16 会退回朴素实现)。")
    print("  - 特征误差略大于 FP32 为 backbone INT8 的正常代价;延迟含 H2D/D2H 与同步,纯 GPU 延迟以 C++ cudaEvent 为准。")
    print("  - 更换模型或预处理后需删除 calib.cache,否则读到旧 scale。")


if __name__ == "__main__":
    main()
