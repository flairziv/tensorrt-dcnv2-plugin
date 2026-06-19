#!/usr/bin/env python
"""DCNv2 数值 oracle 生成 + 自定义 DCN 节点 ONNX 导出 + TensorRT 解析失败复现。

本脚本完成四件事:
  1. 以 torchvision.ops.deform_conv2d 作为数值 oracle(其即 DCNv2),保存一组 I/O 作为测试向量;
  2. 用 onnx.helper 构造最小图:DCNv2(input, offset, mask, weight, bias) -> output(domain=dcn);
  3. 用 trt.OnnxParser 解析,预期在 DCNv2 节点处报 "Plugin not found";
  4. 打印 ONNX 节点表,确认 DCNv2 为唯一自定义节点。

关于手工构造 ONNX 而非 torch.onnx.export:新版 torch 使用 dynamo 导出器,不识别
  autograd.Function.symbolic,会直接导出底层 torchvision.deform_conv2d,因无 ONNX 映射而失败。
  手工构造可精确控制自定义算子契约,且不受导出器版本影响;oracle 仍由 torchvision 计算。
  本图仅含该节点(x/offset/mask 为输入,weight/bias 为常量),上游产生 offset/mask 的 Conv 在端到端模型中引入。

产物:dcn.onnx 与 dcn_io.npz(x/offset/mask/weight/bias/y_oracle 及超参),供 kernel/plugin 数值比对。

运行(WSL,dl env):  python 01_oracle_and_export.py
"""
import numpy as np                       # 保存 .npz / 写裸 .bin / torch-numpy 互转
import onnx                              # 构造计算图(make_node/graph/model)与读写 .onnx
import torch                             # 计算 oracle(deform_conv2d)
import torch.nn as nn                    # nn.Module / nn.Conv2d / nn.Parameter
from onnx import TensorProto, helper, numpy_helper  # 张量 dtype 枚举;构造 node/graph/model;numpy 与 onnx 张量互转

try:
    from torchvision.ops import deform_conv2d   # torchvision 可变形卷积(带 mask 调制,即 DCNv2),用作数值 oracle
except ImportError:
    raise SystemExit("缺少 torchvision,请安装与 torch 匹配的版本:pip install torchvision")

N, CIN, COUT, K = 1, 16, 16, 3   # batch / 输入通道 / 输出通道 / 卷积核边长(3 即 3x3)
H = W = 32                       # 输入特征图高与宽
STRIDE, PAD, DIL = 1, 1, 1       # 卷积超参:步长 / padding / 膨胀率
OFF_CH = 2 * K * K      # 18:offset 通道数 = K*K 个采样点 x 每点 2 个偏移分量(纵向 + 横向)
MASK_CH = K * K         # 9:mask 通道数 = 每个采样点 1 个调制系数(DCNv2 相对 v1 增加的一路)


class DCNv2Block(nn.Module):
    """DCNv2 单元:普通 Conv 产生 offset 与 mask,再执行可变形卷积。"""

    def __init__(self):
        super().__init__()
        self.offset_mask = nn.Conv2d(CIN, OFF_CH + MASK_CH, K, stride=STRIDE, padding=PAD)  # 由 x 产生 offset(18)+ mask(9)= 27 通道
        self.weight = nn.Parameter(torch.randn(COUT, CIN, K, K) * 0.1)  # 可变形卷积权重 [Cout,Cin,K,K]
        self.bias = nn.Parameter(torch.zeros(COUT))                     # 可变形卷积偏置 [Cout]

    def split(self, x):
        om = self.offset_mask(x)               # [N,27,H,W]
        return om[:, :OFF_CH], torch.sigmoid(om[:, OFF_CH:])   # 前 18 通道为 offset;后 9 通道经 sigmoid 得 mask in (0,1)


def main():
    torch.manual_seed(0)              # 固定随机种子,保证 oracle 可复现
    model = DCNv2Block().eval()       # eval 推理模式
    x = torch.randn(N, CIN, H, W)     # 随机输入 [1,16,32,32](NCHW)

    # 1) 计算 oracle 并保存 I/O(后续插件输出须与此 y 对齐)。
    with torch.no_grad():
        offset, mask = model.split(x) # offset[1,18,32,32],mask[1,9,32,32]
        y = deform_conv2d(x, offset, model.weight, model.bias,   # torchvision 计算参考输出 y
                          stride=(STRIDE, STRIDE), padding=(PAD, PAD),
                          dilation=(DIL, DIL), mask=mask)              # 传入 mask 即 DCNv2(非 v1)
    print(f"[oracle] x{tuple(x.shape)} -> y{tuple(y.shape)}  y.sum={y.sum().item():.4f}")
    xn, on, mn = x.numpy(), offset.numpy(), mask.numpy()
    wn, bn, yn = model.weight.detach().numpy(), model.bias.detach().numpy(), y.numpy()  # Parameter 须先 detach 再转 numpy
    np.savez("dcn_io.npz", x=xn, offset=on, mask=mn, weight=wn, bias=bn, y=yn,  # 全部 I/O 打包存入 .npz
             stride=STRIDE, pad=PAD, dilation=DIL, kernel=K)                    # 一并保存超参
    print("[oracle] 已保存 dcn_io.npz")
    for nm, arr in [("x", xn), ("offset", on), ("mask", mn), ("weight", wn), ("bias", bn), ("y", yn)]:
        np.ascontiguousarray(arr, dtype=np.float32).tofile(f"dcn_{nm}.bin")  # 写为 C 连续 float32 裸 .bin,供 C++ 直接 fread
    print("[oracle] 已保存 dcn_{x,offset,mask,weight,bias,y}.bin(供 C++ fread 验证 kernel)")

    # 2) 构造 ONNX(DCNv2 为唯一自定义节点;weight/bias 为常量 initializer)。
    node = helper.make_node(           # 构造 ONNX 节点,即自定义算子契约
        "DCNv2", inputs=["input", "offset", "mask", "weight", "bias"], outputs=["output"],  # op_type 与 5 个输入名(顺序即位置)
        domain="dcn", stride=STRIDE, padding=PAD, dilation=DIL, kernel=K, deformable_groups=1)  # 自定义域 + 5 个属性(由 plugin Creator 读取)
    graph = helper.make_graph(
        [node], "dcnv2_graph",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, list(xn.shape)),
                helper.make_tensor_value_info("offset", TensorProto.FLOAT, list(on.shape)),
                helper.make_tensor_value_info("mask", TensorProto.FLOAT, list(mn.shape))],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(yn.shape))],
        initializer=[numpy_helper.from_array(wn, "weight"), numpy_helper.from_array(bn, "bias")])  # weight/bias 内嵌为常量
    onnx.save(helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("dcn", 1)]), "dcn.onnx")  # 标准域 opset 18,自定义域 dcn 版本 1
    print("[onnx] 已构造 dcn.onnx:DCNv2(input, offset, mask + weight/bias[常量]) -> output")

    # 3) 打印节点表(确认 DCNv2 为唯一自定义节点)。
    print("[onnx] 节点 (op_type @ domain):")
    for nd in onnx.load("dcn.onnx").graph.node:
        dom = nd.domain if nd.domain else "(标准)"  # 空 domain 为标准算子,否则为自定义域
        flag = "   (自定义,需提供 plugin)" if nd.domain == "dcn" else ""
        print(f"    {nd.op_type:14s} @ {dom}{flag}")

    # 4) TensorRT 解析,预期在 DCNv2 处失败。
    import tensorrt as trt              # 延迟 import:前序步骤不依赖 TRT
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):  # 版本兼容判断
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open("dcn.onnx", "rb") as f:
        ok = parser.parse(f.read())          # 解析结果:True 成功 / False 失败
    print(f"\n[parse] parser.parse() = {ok}   (预期 False:TRT 无 DCNv2 的 importer)")
    for i in range(parser.num_errors):
        print(f"    #{i}: {parser.get_error(i).desc()}")  # 预期包含 DCNv2 的 "Plugin not found"
    if not ok:
        print("\n解析在自定义节点处失败,符合预期。")
        print("  下一步:实现 DCN CUDA kernel 与 DCNv2 的 IPluginV3(名称 'DCNv2',5 个输入),")
        print("          使解析通过,且输出与 dcn_io.npz 中的 y 对齐。")


if __name__ == "__main__":
    main()
