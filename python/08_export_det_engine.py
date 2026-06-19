#!/usr/bin/env python
"""导出含检测头的完整 RetinaNet 至 TRT 引擎,并产出 C++ 端到端检测所需的全部辅助文件。

05/07 仅将 backbone 放入引擎(输出 FPN 特征图)。为实现纯 C++ 出框,将检测头一并放入引擎,
使引擎直接输出每个 anchor 的 cls_logits[A,K] 与 bbox_reg[A,4];C++ 仅做 anchor 解码与 class-aware NMS。

产物(均供 C++ 使用):
  det.engine        含检测头的引擎(DCN 走 plugin FP32,其余 FP16)
  anchors.bin       [A,4] float32,由 torchvision AnchorGenerator 生成,C++ 直接读取
  det_input.bin     [1,3,512,512] float32,固定输入(供 C++ 运行与对齐)
  det_meta.txt      尺寸/类数/各层 anchor 数/阈值 等标量(简单 KV)
  det_categories.txt COCO 类名(每行一个)
  det_ref.txt       numpy 参考框(N 行:label score x1 y1 x2 y2),供 C++ 对齐

运行(WSL,dl env,需 ../src/build/libdcnv2.so):  python 08_export_det_engine.py
"""
import ctypes
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt
from torchvision.io import read_image                                       # 读图(uint8 CHW)
from torchvision.transforms.functional import convert_image_dtype, normalize, resize  # 预处理
from torchvision.models.detection.image_list import ImageList   # 为 AnchorGenerator 包装输入尺寸

from detector import build_dcn_detector

PLUGIN = "../src/build/libdcnv2.so"
SIZE = 512                       # 固定输入边长(32 的倍数,适配 FPN)
CLIP = math.log(1000.0 / 16.0)   # box 回归 dw/dh 上限(同 torchvision BoxCoder,防 exp 溢出)


class HeadForward(nn.Module):
    """仅保留 backbone + head 的前向:输出 (cls_logits[1,A,K], bbox_reg[1,A,4])。
    去除 torchvision 的 transform / anchor / postprocess(分别放入 C++ 或预生成)。"""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone     # ResNet50 + FPN(含一层 DCN)
        self.head = model.head             # RetinaNetHead(分类子网 + 回归子网)

    def forward(self, x):
        feats = list(self.backbone(x).values())   # 5 张 FPN 特征图(顺序即层序)
        h = self.head(feats)                       # head 内部对各层 permute + reshape 后 concat
        return h["cls_logits"], h["bbox_regression"]   # [1,A,K] logits(未 sigmoid),[1,A,4] 回归量


# 与 C++ 一一对应的 numpy 后处理(作为对齐参考)。
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _decode(reg, anc):
    """anchor + 回归量 -> 框 [x1,y1,x2,y2](torchvision BoxCoder,weights=1)。"""
    w = anc[:, 2] - anc[:, 0]
    h = anc[:, 3] - anc[:, 1]
    cx = anc[:, 0] + 0.5 * w
    cy = anc[:, 1] + 0.5 * h
    dx, dy, dw, dh = reg[:, 0], reg[:, 1], reg[:, 2], reg[:, 3]
    dw = np.minimum(dw, CLIP)            # 限幅,防 exp 溢出
    dh = np.minimum(dh, CLIP)
    pcx = dx * w + cx                    # 预测中心 x
    pcy = dy * h + cy                    # 预测中心 y
    pw = np.exp(dw) * w                  # 预测宽
    ph = np.exp(dh) * h                  # 预测高
    return np.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], axis=1)


def _iou(a, b):
    """单框 a 与一批框 b 的 IoU。"""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def _nms_classaware(boxes, scores, labels, thr):
    """class-aware 贪心 NMS:仅抑制同类且 IoU>thr 的低分框。返回按分数降序的保留下标。"""
    order = np.argsort(-scores, kind="stable")   # 分数降序(stable 以对齐 C++ std::stable_sort)
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for ii in range(len(order)):
        if suppressed[ii]:
            continue
        i = order[ii]
        keep.append(i)
        for jj in range(ii + 1, len(order)):
            if suppressed[jj]:
                continue
            j = order[jj]
            if labels[i] == labels[j] and _iou(boxes[i], boxes[j:j + 1])[0] > thr:   # 仅比较同类
                suppressed[jj] = True
    return keep


def postprocess(cls, reg, anchors, HWA, K, score_thresh, topk, nms_thresh, det_per_img, size):
    """逐层 阈值+topk -> 解码 -> clip -> 跨层 class-aware NMS -> top-N。返回 (boxes, scores, labels)。"""
    B, S, L = [], [], []
    off = 0
    for hwa in HWA:                                  # 逐 FPN 层(顺序与 head/anchors 一致)
        c = cls[off:off + hwa]                       # [hwa,K] logits
        r = reg[off:off + hwa]                       # [hwa,4]
        a = anchors[off:off + hwa]                   # [hwa,4]
        s = _sigmoid(c).reshape(-1)                  # [hwa*K] 概率
        kept = np.where(s > score_thresh)[0]         # 过阈值的展平下标
        if kept.size:
            nt = min(topk, kept.size)                # 每层最多取 topk 个候选
            top = kept[np.argsort(-s[kept], kind="stable")[:nt]]
            anc_i = top // K                         # 候选对应的 anchor 下标(层内)
            lab = top % K                            # 类别
            B.append(_decode(r[anc_i], a[anc_i]))    # 解码框
            S.append(s[top]); L.append(lab)
        off += hwa
    if not B:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
    boxes = np.concatenate(B); scores = np.concatenate(S); labels = np.concatenate(L)
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, size)    # clip 至图像范围
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, size)
    keep = _nms_classaware(boxes, scores, labels, nms_thresh)[:det_per_img]   # NMS 并限制总数
    return boxes[keep], scores[keep], labels[keep]


def build_engine(logger):
    """解析 det.onnx -> FP16 引擎(DCN plugin 层强制 FP32 走快速路径)-> 字节流。"""
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) \
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") else 0
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open("det.onnx", "rb").read()):
        for i in range(parser.num_errors):
            print("   ", parser.get_error(i).desc())
        raise SystemExit("parse 失败")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    ptypes = [getattr(trt.LayerType, n) for n in ("PLUGIN_V3", "PLUGIN_V2", "PLUGIN")
              if hasattr(trt.LayerType, n)]
    for i in range(net.num_layers):
        if net.get_layer(i).type in ptypes:
            try:
                net.get_layer(i).precision = trt.DataType.FLOAT   # DCN 走 FP32 快速路径
            except Exception:  # noqa: BLE001
                pass
    print("[trt] 正在构建 det 引擎(backbone + head,约 30-60s)...")
    return bytes(builder.build_serialized_network(net, cfg))


def make_input(img_path):
    """读真实图 -> resize 到 512*512 -> [0,1] -> ImageNet 归一化(对齐 RetinaNet 的 transform)-> [1,3,512,512]。
    引擎为 HeadForward(已去除 torchvision 的 transform),故输入必须为已归一化的张量。"""
    img = read_image(img_path)[:3]                              # uint8 CHW(去 alpha)
    img = resize(img, [SIZE, SIZE], antialias=True)            # 缩放至固定尺寸(引擎为定尺寸输入)
    x = convert_image_dtype(img, torch.float)                  # -> [0,1]
    x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 减均值除标准差,与 torchvision 归一化一致
    return x.unsqueeze(0)                                       # 添加 batch 维


def main():
    model, cats = build_dcn_detector()              # 完整 RetinaNet(FPN 含一层 DCN)
    fwd = HeadForward(model).eval()
    img_path = sys.argv[1] if len(sys.argv) > 1 else None       # 可选:命令行传入一张真实图片
    if img_path:
        x = make_input(img_path)                                # 真实图才有真实检测,对齐才有意义
        print(f"[input] 使用真实图 {img_path}(resize 到 {SIZE}*{SIZE})")
    else:
        x = torch.randn(1, 3, SIZE, SIZE)                       # 随机噪声下训练好的 head 多半输出 0 框
        print("[input] 使用随机输入,多半输出 0 框(对齐无意义)。"
              "传入图片以获得真实检测:python 08_export_det_engine.py image.jpg")

    with torch.no_grad():
        feats = list(model.backbone(x).values())    # FPN 特征(用于 anchor 生成)
        anchors = model.anchor_generator(ImageList(x, [(SIZE, SIZE)]), feats)[0]  # [A,4] 输入像素坐标
        cls, reg = fwd(x)                            # [1,A,K],[1,A,4]
    A, K = anchors.shape[0], cls.shape[2]
    nA = model.anchor_generator.num_anchors_per_location()[0]   # 每位置 anchor 数(=9)
    HWA = [int(f.shape[2] * f.shape[3] * nA) for f in feats]    # 各层 anchor 数(顺序与 concat 一致)
    print(f"[info] A={A} K={K} 每层 anchor 数={HWA} sum={sum(HWA)}(应 == A)")

    # 导出 ONNX 并构建引擎。
    torch.onnx.export(fwd, x, "det.onnx", input_names=["input"], output_names=["cls", "reg"],
                      opset_version=17, custom_opsets={"dcn": 1}, dynamo=False)   # dynamo=False 才包含 dcn::DCNv2
    ctypes.CDLL(PLUGIN)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    plan = build_engine(logger)
    open("det.engine", "wb").write(plan)
    print(f"[trt] det.engine 已生成:{len(plan) / 1e6:.1f} MB")

    # 保存 C++ 辅助文件。
    anchors.numpy().astype(np.float32).tofile("anchors.bin")
    x.numpy().astype(np.float32).tofile("det_input.bin")
    with open("det_meta.txt", "w") as f:            # 简单 KV,便于 C++ 解析
        f.write(f"size {SIZE}\nnum_classes {K}\nnum_anchors {A}\n")
        f.write(f"score_thresh {model.score_thresh}\nnms_thresh {model.nms_thresh}\n")
        f.write(f"topk {model.topk_candidates}\ndet_per_img {model.detections_per_img}\n")
        f.write("hwa " + " ".join(str(v) for v in HWA) + "\n")
    with open("det_categories.txt", "w") as f:
        f.write("\n".join(cats))

    # 生成 numpy 参考框(对齐参考)。
    # 使用引擎输出(FP16 计算)而非 PyTorch 的 FP32 输出来生成参考,使其与 C++(同一引擎)仅差"后处理实现",
    # 从而隔离 FP16/FP32 数值差异,使 parity 干净地验证 C++ 解码 + NMS 的正确性。
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    with TrtRunner(EngineFromBytes(plan)) as r:
        eo = r.infer({"input": np.ascontiguousarray(x.numpy(), np.float32)})   # 运行引擎(plugin 已 CDLL 注册)
    cn = np.array(eo["cls"], np.float32).reshape(A, K)     # 引擎输出的 cls_logits [A,K]
    rn = np.array(eo["reg"], np.float32).reshape(A, 4)     # 引擎输出的 bbox_reg [A,4]
    an = anchors.numpy()
    print(f"[info] 引擎输出 max sigmoid score = {float(_sigmoid(cn).max()):.4f}"
          f"(随机输入下很小,趋于 0 框;真实图通常 0.3~0.99)")
    boxes, scores, labels = postprocess(cn, rn, an, HWA, K, model.score_thresh,
                                        model.topk_candidates, model.nms_thresh,
                                        model.detections_per_img, SIZE)
    order = np.argsort(-scores, kind="stable")      # 按分数降序写出(与 C++ 输出同序,便于逐行比对)
    with open("det_ref.txt", "w") as f:
        f.write(f"{len(order)}\n")
        for i in order:
            f.write(f"{int(labels[i])} {scores[i]:.6f} "
                    f"{boxes[i][0]:.4f} {boxes[i][1]:.4f} {boxes[i][2]:.4f} {boxes[i][3]:.4f}\n")
    print(f"[ref] numpy 参考框 {len(order)} 个 -> det_ref.txt(供 C++ 对齐)")
    print("\n下一步:cd ../cpp && cmake --build build -> ./build/detect -> 与 det_ref.txt 对齐验证。")


if __name__ == "__main__":
    main()
