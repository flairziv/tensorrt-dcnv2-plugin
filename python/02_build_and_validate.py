#!/usr/bin/env python
"""加载 DCNv2 plugin -> 解析 dcn.onnx(此次应成功)-> 构建引擎 -> 推理 -> 与 oracle 比对。

闭环:01 生成的 dcn.onnx 此前因 "Plugin not found" 解析失败;本脚本先加载 libdcnv2.so(Creator 入注册表),
  再解析(应通过)-> 构建引擎 -> 输入 dcn_io.npz 的 input/offset/mask(weight/bias 已内嵌为常量)->
  输出与 oracle 的 y 对齐。kernel 已单独验证(test_dcn max|err|=3.6e-7),本脚本验证的是 plugin 接线。

运行(WSL,dl env;需先 cmake 编出 ../src/build/libdcnv2.so):
  python 02_build_and_validate.py
"""
import ctypes              # 通过 CDLL 加载插件 .so(触发其注册宏)
import os                  # 检查文件是否存在

import numpy as np         # 读 npz 与数值比对
import tensorrt as trt     # 先 import 使 libnvinfer 进入进程,满足 CDLL libdcnv2 的依赖

PLUGIN = "../src/build/libdcnv2.so"   # 编译产出的插件动态库
ONNX = "dcn.onnx"                      # 01 构造的含 DCNv2 节点的图
NPZ = "dcn_io.npz"                     # 01 保存的 oracle I/O(含参考答案 y)


def main():
    if not os.path.exists(PLUGIN):                  # 缺插件:需先在 src/ 编译
        raise SystemExit(f"缺少 {PLUGIN},请先在 src/ 用 cmake 编译插件")
    for p in (ONNX, NPZ):                           # 缺 onnx/npz:需先运行 01
        if not os.path.exists(p):
            raise SystemExit(f"缺少 {p},请先运行 01_oracle_and_export.py 生成")

    ctypes.CDLL(PLUGIN)                          # 触发 REGISTER_TENSORRT_PLUGIN,DCNv2 Creator 入注册表
    print(f"[1/4] 已加载 {PLUGIN}")
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")      # 初始化 TRT 自带插件库

    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):  # 版本兼容判断
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(ONNX, "rb") as f:
        ok = parser.parse(f.read())              # 解析(注册表已含 DCNv2,应成功)
    print(f"[2/4] parser.parse() = {ok}   (此前为 False,此次应为 True)")
    if not ok:
        for i in range(parser.num_errors):
            print("    ", parser.get_error(i).desc())
        raise SystemExit("parse 仍失败:.so 未加载 / name、version、namespace 不一致")
    names = [network.get_input(i).name for i in range(network.num_inputs)]  # 运行时输入名(weight/bias 为常量,不在此列)
    print(f"      图输入(weight/bias 为常量,不在此列):{names}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # workspace 上限 1GiB
    plan = builder.build_serialized_network(network, config)  # 构建并序列化引擎(此处调用 plugin 的 getOutputShapes / supportsFormatCombination)
    if plan is None:
        raise SystemExit("build 失败(检查 getOutputShapes / supportsFormatCombination)")
    plan = bytes(plan)
    print(f"[3/4] 引擎构建成功:{len(plan) / 1e3:.1f} KB")

    d = np.load(NPZ)
    src = {"input": d["x"], "offset": d["offset"], "mask": d["mask"]}    # 仅喂图的 3 个输入
    feeds = {n: np.ascontiguousarray(src[n], dtype=np.float32) for n in names}  # 按输入名组织数据(连续 float32)
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    with TrtRunner(EngineFromBytes(plan)) as runner:
        out = runner.infer(feeds)                # 推理,返回 {输出名: 数组}
    y = np.array(list(out.values())[0], dtype=np.float32).reshape(d["y"].shape)  # 取唯一输出并 reshape 为 oracle 形状

    err = float(np.abs(y - d["y"]).max())        # 与 oracle 的最大绝对误差
    print(f"[4/4] TRT(DCNv2 plugin) vs torchvision oracle: max|err| = {err:.3e}")
    if err < 1e-3:
        print("\n[PASS] 真实 DCN 算子 -> IPluginV3 插件 -> onnx 解析通过 -> 引擎输出与 oracle 对齐。")
    else:
        print("\n[FAIL] 端到端不一致。kernel 已单独验证,需检查 plugin 接线:")
        print("  输入顺序(0=x/1=offset/2=mask/3=weight/4=bias)、属性名、getOutputShapes 公式。")


if __name__ == "__main__":
    main()
