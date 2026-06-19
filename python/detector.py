#!/usr/bin/env python
"""检测器:预训练 RetinaNet(COCO)+ 在 FPN 中插入一层 DCNv2。

采用 identity 初始化(offset=0,mask 约为 1),使 DCN 行为近似原 3x3 conv,检测效果保持不变,
因此无需训练即可用于部署演示。该 DCN 以 DCNv2Fn(自定义算子)实现,既可在 PyTorch 运行,
又可经 dynamo=False 导出为 dcn::DCNv2 节点供 TRT plugin 使用。
"""
from dataclasses import dataclass        # 轻量结构体(封装检测结果)

import numpy as np                        # 支持 ndarray 输入
import torch                              # PyTorch
import torch.nn as nn                     # 网络层 / 模块
import torchvision                        # 预训练检测模型(RetinaNet)
from torchvision.io import read_image, write_jpeg            # 读图(uint8 CHW)/ 写 jpg
from torchvision.ops import deform_conv2d # 可变形卷积前向(DCNv2 计算)
from torchvision.transforms.functional import convert_image_dtype  # uint8 -> float[0,1]
from torchvision.utils import draw_bounding_boxes            # 绘制检测框

K = 3                       # 卷积核边长 3x3
OFF, MSK = 2 * K * K, K * K # OFF=offset 通道数(18);MSK=mask 通道数(9)


# 自定义算子封装:使一次 DCN 既能在 PyTorch 前向,又能导出为自定义 ONNX 节点。
class DCNv2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, offset, mask, weight, bias):   # 前向:调用 torchvision 的 deform_conv2d
        return deform_conv2d(x, offset, weight, bias, stride=(1, 1), padding=(1, 1),
                             dilation=(1, 1), mask=mask)

    @staticmethod
    def symbolic(g, x, offset, mask, weight, bias):    # 符号函数:导出 ONNX 时映射为 dcn::DCNv2 节点
        return g.op("dcn::DCNv2", x, offset, mask, weight, bias,         # 节点名 / 输入顺序 / 属性与 plugin 契约一致
                    stride_i=1, padding_i=1, dilation_i=1, kernel_i=K, deformable_groups_i=1)  # _i 后缀表示 int 属性


class DCNBlock(nn.Module):
    """将一个 3x3(cin==cout,stride1)conv 替换为 DCNv2,以 identity 初始化保持原行为。"""

    def __init__(self, conv: nn.Conv2d):               # 传入待替换的原始 3x3 卷积
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone())   # 复用原 conv 权重
        self.bias = nn.Parameter(conv.bias.detach().clone()        # 复用原 conv bias
                                 if conv.bias is not None else torch.zeros(conv.out_channels))  # 原 conv 无 bias 则置 0
        self.om = nn.Conv2d(conv.in_channels, OFF + MSK, K, padding=1)  # 新增卷积产生 offset + mask(27 通道)
        nn.init.zeros_(self.om.weight)                  # offset/mask 卷积权重置 0
        with torch.no_grad():
            self.om.bias.zero_()                        # bias 全置 0,使 offset 恒为 0(采样回到规则网格)
            self.om.bias[OFF:] = 10.0                   # mask logits 取大值,使 sigmoid 约为 1(近似不调制)

    def forward(self, x):
        om = self.om(x)                                 # 产生 offset + mask(offset 约 0、mask 约 1,DCN 近似普通卷积)
        return DCNv2Fn.apply(x, om[:, :OFF], torch.sigmoid(om[:, OFF:]), self.weight, self.bias)  # 拆分后走自定义 DCN 算子


def _replace_first_conv3x3(module) -> bool:
    """在 module 树中查找第一个 3x3、同进出通道、stride1 的 Conv2d 并替换为 DCNBlock(与版本无关的稳健做法)。"""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3) \
                and child.in_channels == child.out_channels and child.stride == (1, 1):  # 命中:3x3 / 同进出通道 / stride1
            setattr(module, name, DCNBlock(child))     # 原地替换为 DCNBlock
            return True                                 # 仅替换一个(只插一层 DCN)
        if _replace_first_conv3x3(child):              # 否则递归继续查找
            return True
    return False


def build_dcn_detector():
    """预训练 RetinaNet,将 FPN 中第一个 3x3 conv 替换为 identity-init DCN。返回 (model, categories)。"""
    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT  # 默认 COCO 预训练权重(含类别表)
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)   # 加载预训练 RetinaNet(ResNet50 + FPN)
    if not _replace_first_conv3x3(model.backbone.fpn.layer_blocks):   # 在 FPN 的 layer_blocks 中插入 DCN
        raise RuntimeError("FPN layer_blocks 中未找到可替换的 3x3 conv(torchvision 版本结构可能已变更)")
    model.eval()
    return model, weights.meta["categories"]            # 返回模型与 COCO 类别名表


# ============================================================================
# 封装:开箱即用的检测器类。
# ============================================================================
@dataclass
class Detection:
    """单条检测结果。"""
    label: str                  # 类别名
    score: float                # 置信度 0~1
    box: tuple                  # 边界框 (x1, y1, x2, y2),像素坐标


class DCNDetector:
    """含 DCN 的 RetinaNet 检测器,封装为一次构建、反复调用。

    用法:
        det = DCNDetector(score_thresh=0.5)          # 构建一次(首次下载约 130MB 权重)
        results = det.detect("photo.jpg")            # -> [Detection(label, score, box), ...]
        det.detect_image("photo.jpg", "out.jpg")     # 检测并绘制存图,返回 results
        print(det.classes)                           # 可识别的 COCO 类名
    """

    def __init__(self, score_thresh: float = 0.5, device: str = None):
        self.score_thresh = score_thresh                          # 置信度阈值(低于则丢弃)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
        self.model, self._categories = build_dcn_detector()        # 构建含 DCN 的检测器并获取 COCO 类别表
        self.model.to(self.device)

    @property
    def classes(self) -> list:
        """可识别的类名(过滤 torchvision 类别表中的占位符 __background__ / N/A)。"""
        return [c for c in self._categories if c not in ("__background__", "N/A")]

    @torch.no_grad()
    def detect(self, image) -> list:
        """对一张图做检测。image 支持:路径 str / np.ndarray(HWC,uint8)/ torch.Tensor(CHW)。
        返回 List[Detection](已按阈值过滤)。"""
        img = self._to_uint8_chw(image)               # 统一为 uint8 CHW 张量
        x = convert_image_dtype(img, torch.float).to(self.device)  # 转 [0,1] float;模型内部自带 resize 与 normalize
        out = self.model([x])[0]                       # 检测器接收 list[图],取第 0 张结果
        keep = out["scores"] > self.score_thresh       # 阈值过滤掩码
        results = []
        for li, sc, bx in zip(out["labels"][keep].tolist(),
                              out["scores"][keep].tolist(),
                              out["boxes"][keep].tolist()):
            results.append(Detection(self._categories[li], float(sc), tuple(bx)))
        return results

    def detect_image(self, image, out_path: str = "det_out.jpg") -> list:
        """检测并将框 / 标签绘制到图上存盘,返回 results。"""
        img = self._to_uint8_chw(image)               # uint8 CHW(绘制需原图)
        results = self.detect(img)
        if results:                                    # 有框才绘制(draw_bounding_boxes 对空框会报错)
            boxes = torch.tensor([r.box for r in results])               # [N,4]
            labels = [f"{r.label} {r.score:.2f}" for r in results]       # "类别 分数"
            img = draw_bounding_boxes(img, boxes, labels, width=3)
        write_jpeg(img, out_path)
        return results

    @staticmethod
    def _to_uint8_chw(image):
        """将 路径 / ndarray(HWC)/ Tensor(CHW)统一为 uint8 CHW 张量。"""
        if isinstance(image, str):
            return read_image(image)[:3]               # 读为 uint8 CHW(去除可能的 alpha 通道)
        if isinstance(image, np.ndarray):
            t = torch.from_numpy(image)
            if t.ndim == 3 and t.shape[2] in (3, 4):   # HWC -> CHW
                t = t.permute(2, 0, 1)
            return t[:3].to(torch.uint8)
        if isinstance(image, torch.Tensor):
            t = image
            if t.dtype != torch.uint8:                 # float[0,1] -> uint8
                t = (t.clamp(0, 1) * 255).to(torch.uint8)
            return t[:3]
        raise TypeError("image 须为 路径 str / np.ndarray(HWC)/ torch.Tensor(CHW)")
