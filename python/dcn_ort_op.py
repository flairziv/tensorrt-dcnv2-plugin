#!/usr/bin/env python
"""向量化 DCNv2 forward + 注册为 onnxruntime custom op(dcn::DCNv2)。

为什么需要它:ONNX flow 的校准(modelopt.onnx)要用 onnxruntime 跑整张图收集激活范围,
  而图里的 dcn::DCNv2 是自定义算子、ORT 没有对应 kernel,InferenceSession 创建即失败。
  这里手写 DCN 的 forward 并注册成 ORT custom op,ORT 遇到 dcn::DCNv2 就回调到本文件的 Python 实现。

数值与 src/dcn_kernel.cu 同语义(已对齐 torchvision oracle);此处用 im2col + GEMM 的向量化写法
  (仅保留 K*K 个窗口点的 Python 循环,Cin/Ho/Wo 全部交给 numpy),backbone 256 通道也跑得动。

整体角色:本文件是"模块",不是测试脚本——被 09_ptq_modelopt_qdq.py import 使用;
  __main__ 里的 _selftest() 只是自检(对齐 01 的 oracle)。三个对外函数:
    ort_ext_lib()        注册 op、返回 extensions 桥接库路径(给 SessionOptions)
    make_session(onnx)   建一个已启用本 op 的 ORT 会话
    rewrite_dcn_domain() 把图里 DCN 节点域改到 ai.onnx.contrib(ORT 能解析的域)

用法(在自己的 ORT 会话里启用本 op):
  import dcn_ort_op, onnxruntime as ort
  so = ort.SessionOptions(); so.register_custom_ops_library(dcn_ort_op.ort_ext_lib())
  sess = ort.InferenceSession("backbone.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

自检(对齐 01 落盘的 oracle):
  python dcn_ort_op.py            # numpy 向量化 vs dcn_io.npz;再(若装了 extensions)ORT 经 custom op vs oracle
"""
import numpy as np

# onnxruntime-extensions 的 Python 自定义算子(PyOp)只能挂在这个域下;若图里 DCN 节点在别的域
# (我们导出时是 "dcn"),ORT 会报 "not a registered op" -> 校准/自检前用 rewrite_dcn_domain 改到这个域。
EXT_DOMAIN = "ai.onnx.contrib"


# ---- 向量化双线性采样:x[Cin,H,W] 在网格 (h,w)[Ho,Wo] 上采样 -> [Cin,Ho,Wo] ----
# 对应 dcn_kernel.cu 的 bilinear<T>:逐角点判越界(界外取 0)+ 中心越界整体取 0,语义与 torchvision 一致。
# 关键:对"所有 Cin 通道、所有输出位置"一次性算完,不再逐元素循环。
def bilinear_grid(x, h, w):
    Cin, H, W = x.shape
    h_low = np.floor(h).astype(np.int64)          # 采样点左上角整数行(向下取整)
    w_low = np.floor(w).astype(np.int64)          # 采样点左上角整数列
    h_high = h_low + 1                            # 右下角整数行
    w_high = w_low + 1                            # 右下角整数列
    lh = (h - h_low).astype(np.float32)           # 到 low 行的小数距离(= 下行的权重分量)
    lw = (w - w_low).astype(np.float32)           # 到 low 列的小数距离(= 右列的权重分量)
    hh = 1.0 - lh                                 # 到 high 行的距离(= 上行的权重分量)
    hw = 1.0 - lw                                 # 到 high 列的距离(= 左列的权重分量)

    def gather(ch, cw, ok):                                   # 取一个角点,界外的位置乘 0
        # 先 clip 坐标到 [0,H-1]/[0,W-1] 避免越界索引(否则 numpy 报错或回绕);
        # 真正"界外"的位置由布尔 ok 乘 0 抹掉,等价 kernel 里逐角点的 if 判断。
        g = x[:, np.clip(ch, 0, H - 1), np.clip(cw, 0, W - 1)]   # 花式索引一次取全 Cin 通道 -> [Cin,Ho,Wo]
        return g * ok[None]                                  # ok[Ho,Wo] 广播到 [1,Ho,Wo]

    g_ll = gather(h_low, w_low, (h_low >= 0) & (w_low >= 0))             # 左上角
    g_lh = gather(h_low, w_high, (h_low >= 0) & (w_high <= W - 1))       # 右上角
    g_hl = gather(h_high, w_low, (h_high <= H - 1) & (w_low >= 0))       # 左下角
    g_hh = gather(h_high, w_high, (h_high <= H - 1) & (w_high <= W - 1)) # 右下角
    # 双线性 = 四角按"对角面积"加权(行权重 hh/lh × 列权重 hw/lw)
    out = ((hh * hw)[None] * g_ll + (hh * lw)[None] * g_lh
           + (lh * hw)[None] * g_hl + (lh * lw)[None] * g_hh)
    valid = ((h > -1) & (h < H) & (w > -1) & (w < W)).astype(np.float32)   # 采样中心整体越界则取 0
    return out * valid[None]


# ---- 向量化 DCNv2 前向(im2col + GEMM)----
# 与 dcn_kernel.cu 的快速路径同构:每个窗口点采样一次填入 cols[Cin*K*K, Ho*Wo],再 weight @ cols。
# 输入约定(与 plugin/01 一致):
#   x[N,Cin,H,W]  offset[N,2*K*K,Ho,Wo]  mask[N,K*K,Ho,Wo]  weight[Cout,Cin,K,K]  bias[Cout]
#   offset 通道排布:每个窗口点 p 占 2 通道(2p=纵向 dh,2p+1=横向 dw);mask 每点 1 通道。
def dcn_v2_forward(x, offset, mask, weight, bias, stride=1, pad=1, dil=1):
    x = np.asarray(x, np.float32)                 # 统一 float32(ORT 传进来的可能是别的视图)
    offset = np.asarray(offset, np.float32)
    mask = np.asarray(mask, np.float32)
    weight = np.asarray(weight, np.float32)
    bias = np.asarray(bias, np.float32)
    N, Cin, H, W = x.shape
    Cout, _, K, _ = weight.shape
    Ho, Wo = offset.shape[2], offset.shape[3]                # 输出空间尺寸(由 offset/mask 给定)
    oh = np.arange(Ho, dtype=np.float32)[:, None]            # 规则网格行坐标 [Ho,1]
    ow = np.arange(Wo, dtype=np.float32)[None, :]            # 规则网格列坐标 [1,Wo]
    wmat = weight.reshape(Cout, Cin * K * K)                 # 权重摊平 [Cout, CKK];列序 = ic*K*K + i*K + j
    y = np.empty((N, Cout, Ho, Wo), np.float32)
    for n in range(N):                                       # 逐张图(本项目 N=1,循环只为通用)
        cols = np.empty((Cin * K * K, Ho, Wo), np.float32)   # im2col 矩阵:每行 =(ic,i,j)在各输出位置的调制采样值
        for i in range(K):                                   # 卷积窗口行
            for j in range(K):                               # 卷积窗口列(仅这 K*K 次 Python 循环)
                p = i * K + j                                # 窗口内采样点序号
                h_im = (oh * stride - pad + i * dil) + offset[n, 2 * p]      # [Ho,Wo] 实际采样行 = 规则网格 + 偏移
                w_im = (ow * stride - pad + j * dil) + offset[n, 2 * p + 1]  # [Ho,Wo] 实际采样列
                samp = bilinear_grid(x[n], h_im, w_im) * mask[n, p][None]    # [Cin,Ho,Wo] 双线性采样后乘 mask 调制
                cols[p::K * K] = samp                        # 写入行 ic*K*K + p(切片 p::K*K 恰好选中这些行)
        # 一发 GEMM 出所有输出通道(消除朴素实现按 Cout 重复采样的冗余),再加 bias
        y[n] = (wmat @ cols.reshape(Cin * K * K, Ho * Wo)).reshape(Cout, Ho, Wo) + bias[:, None, None]
    return y


# ---- 注册为 onnxruntime custom op:dcn::DCNv2 ----
_REGISTERED = False                                          # 防止重复注册(@onnx_op 同名重复会报错)


def _register():
    """用 @onnx_op 把上面的 forward 绑到 ai.onnx.contrib::DCNv2(import 时调用一次即可)。

    注意:onnxruntime-extensions 的 PyOp 只能挂在 ai.onnx.contrib 域,自定义域 "dcn" 不被 PyOp 识别;
    所以要运行的图必须先经 rewrite_dcn_domain 把 DCN 节点域改成 EXT_DOMAIN。
    """
    global _REGISTERED
    if _REGISTERED:                                          # 已注册过直接返回(可被多次调用)
        return
    from onnxruntime_extensions import onnx_op, PyCustomOpDef  # 延迟 import:没装 extensions 时不影响 numpy forward

    # 本项目 DCN 固定 stride=pad=dil=1、K=3(见 detector.py 的 symbolic),故 forward 直接用这组超参;
    # 不在装饰器声明 attrs,避开各版本 extensions 的属性传递差异。若你的 DCN 超参不同,再改这里。
    @onnx_op(op_type="DCNv2", domain=EXT_DOMAIN,             # 绑定 (域, 算子名);需与图里节点一致
             inputs=[PyCustomOpDef.dt_float] * 5,            # 5 个输入:x, offset, mask, weight, bias(均 float)
             outputs=[PyCustomOpDef.dt_float])               # 1 个输出
    def _dcnv2(x, offset, mask, weight, bias):               # ORT 执行到该节点时回调此函数(weight/bias 来自 initializer)
        return dcn_v2_forward(x, offset, mask, weight, bias, stride=1, pad=1, dil=1)

    _REGISTERED = True


def ort_ext_lib():
    """注册 op 并返回 onnxruntime-extensions 的桥接库路径(传给 SessionOptions.register_custom_ops_library)。"""
    _register()                                              # 确保 @onnx_op 已执行(把 forward 入注册表)
    from onnxruntime_extensions import get_library_path
    return get_library_path()                                # 这个 .so 是 ORT↔Python 的桥;ORT 加载它后才能回调 PyOp


def make_session(onnx_path, providers=None):
    """便捷:建一个已启用 DCNv2 custom op 的 InferenceSession(onnx_path 的 DCN 节点须在 EXT_DOMAIN 域)。"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.register_custom_ops_library(ort_ext_lib())            # 按会话注册:ORT 这个 session 才认得 DCNv2
    providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]  # PyOp 在 CPU 跑,其余层可在 GPU
    return ort.InferenceSession(onnx_path, so, providers=providers)


def rewrite_dcn_domain(src, dst, to_domain=EXT_DOMAIN):
    """把 ONNX 里 DCNv2 节点的 domain 改成 ORT custom op 所在域(默认 ai.onnx.contrib)并补 opset_import。
    仅用于让 ORT 校准/自检能解析 DCN;交给 TRT 的图仍保留原 "dcn" 域(plugin 按 op_type 匹配,见 02)。"""
    import onnx
    m = onnx.load(src)
    n = 0
    for node in m.graph.node:                                # 遍历所有节点,改 DCNv2 的 domain
        if node.op_type == "DCNv2":
            node.domain = to_domain
            n += 1
    if to_domain not in {op.domain for op in m.opset_import}:  # 新域要在 opset_import 里声明,否则 ORT/检查器报错
        m.opset_import.append(onnx.helper.make_opsetid(to_domain, 1))
    onnx.save(m, dst)
    print(f"[rewrite] {src} -> {dst}: {n} 个 DCNv2 节点 domain 改为 {to_domain}")
    return dst                                               # 返回 dst 方便链式调用


# ---- 自检:对齐 01 的 oracle ----
def _selftest():
    d = np.load("dcn_io.npz")                                # 01_oracle_and_export.py 落盘的 I/O(含参考答案 y)
    # (1) 纯 numpy:向量化 forward 直接对 oracle,验证 DCN 数学写对了
    y = dcn_v2_forward(d["x"], d["offset"], d["mask"], d["weight"], d["bias"],
                       stride=int(d["stride"]), pad=int(d["pad"]), dil=int(d["dilation"]))
    print("[numpy] 向量化 forward vs torchvision oracle: max|err| =", float(np.abs(y - d["y"]).max()))

    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print("[ort] 未装 onnxruntime,跳过 ORT 自检")
        return
    try:
        # (2) 经 ORT:把 dcn.onnx 的域改到 contrib -> 建带 custom op 的会话 -> 跑图,对 oracle
        ort_onnx = rewrite_dcn_domain("dcn.onnx", "dcn_contrib.onnx")   # DCN 域 dcn -> ai.onnx.contrib 供 ORT 解析
        sess = make_session(ort_onnx, providers=["CPUExecutionProvider"])
        feeds = {"input": d["x"].astype(np.float32),
                 "offset": d["offset"].astype(np.float32),
                 "mask": d["mask"].astype(np.float32)}        # weight/bias 是 initializer,不用传入
        yo = sess.run(None, feeds)[0]
        print("[ort] dcn(custom op)vs oracle: max|err| =", float(np.abs(yo - d["y"]).max()))
    except Exception as exc:  # noqa: BLE001
        print("[ort] 自检失败:", exc)


if __name__ == "__main__":
    _selftest()
