#!/usr/bin/env python
"""探查:含 DCN 的完整 torch 模型能否导出为带 dcn::DCNv2 节点的 ONNX。

完整模型无法像孤立节点那样手工构造,只能依赖导出器。新版 dynamo 导出器忽略
autograd.Function.symbolic,故此处测试旧 TorchScript 导出器(dynamo=False)能否使用 symbolic:
  可行 -> 检测器走"模型内插 DCNv2Fn + export"路线;
  不可行 -> 改用 onnx-graphsurgeon 图手术(导出无 DCN 版本后手动接入 DCNv2 节点)。

运行(WSL,dl env):  python probe_export.py
"""
import torch                              # PyTorch
import torch.nn as nn                     # 网络层
import onnx                               # 读回导出的 onnx 检查节点
from torchvision.ops import deform_conv2d # DCN 前向

K = 3                        # 核大小
OFF, MSK = 2 * K * K, K * K  # offset / mask 通道数


# 自定义算子(与 detector.py 一致,此处单独保留一份便于独立探查)。
class DCNv2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, offset, mask, weight, bias):   # 前向即 deform_conv2d
        return deform_conv2d(x, offset, weight, bias, stride=(1, 1), padding=(1, 1),
                             dilation=(1, 1), mask=mask)

    @staticmethod
    def symbolic(g, x, offset, mask, weight, bias):    # 导出时映射为 dcn::DCNv2 节点
        return g.op("dcn::DCNv2", x, offset, mask, weight, bias,
                    stride_i=1, padding_i=1, dilation_i=1, kernel_i=K, deformable_groups_i=1)


# 最小测试模型:一个 conv 产生 offset/mask + 一层 DCN。
class M(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.om = nn.Conv2d(c, OFF + MSK, K, padding=1)   # 产生 offset + mask
        self.w = nn.Parameter(torch.randn(c, c, K, K) * 0.1)  # DCN 权重
        self.b = nn.Parameter(torch.zeros(c))                 # DCN bias

    def forward(self, x):
        om = self.om(x)                                  # 产生 offset + mask
        return DCNv2Fn.apply(x, om[:, :OFF], torch.sigmoid(om[:, OFF:]), self.w, self.b)  # 走自定义算子


def main():
    m = M().eval()
    x = torch.randn(1, 16, 32, 32)
    try:
        torch.onnx.export(m, x, "probe.onnx", input_names=["input"], output_names=["output"],
                          opset_version=17, custom_opsets={"dcn": 1}, dynamo=False)  # dynamo=False 用旧导出器以识别 symbolic;custom_opsets 声明 dcn 域版本
        nodes = [(n.op_type, n.domain or "(标准)") for n in onnx.load("probe.onnx").graph.node]
        print("[probe] 节点:", nodes)
        if any(d == "dcn" for _, d in nodes):            # 出现 dcn 域节点表示 symbolic 生效
            print("\n旧导出器(dynamo=False)可发出 dcn::DCNv2,检测器采用模型内插 DCNv2Fn + export 路线。")
        else:                                            # 导出成功但无 dcn 节点表示 symbolic 未生效
            print("\n导出成功但无 dcn 节点(symbolic 未生效),改用 onnx-graphsurgeon 图手术。")
    except Exception as e:  # noqa: BLE001
        print(f"\n[probe] dynamo=False 失败:{type(e).__name__}: {e}")
        print("改用 onnx-graphsurgeon 图手术:导出无 DCN 版本后手动接入 DCNv2 节点。")


if __name__ == "__main__":
    main()
