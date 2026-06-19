#!/usr/bin/env python
"""含 DCN 的 RetinaNet 检测器在 PyTorch 上的推理示例(检测并绘制结果框)。

用于验证插入 DCN(identity 初始化)的检测器仍能正常检测,该网络即后续部署到 TRT 插件的目标。
本脚本演示封装类 DCNDetector 的用法。

运行(WSL,dl env;首次会自动下载 RetinaNet 权重约 130MB):
  python 04_detect_image.py image.jpg
"""
import sys                                  # 读取命令行参数(图片路径)

from detector import DCNDetector            # 封装的检测器类(构建/检测/绘制/类别清单)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"   # 第 1 参数为图片路径(缺省 test.jpg)

    det = DCNDetector(score_thresh=0.5)      # 构建含 DCN 的 RetinaNet(自动选择 CUDA/CPU)
    try:
        results = det.detect_image(img_path, "det_out.jpg")   # 检测并绘制存图,返回结果列表
    except Exception:
        raise SystemExit(f"无法读取图片 {img_path}。用法:python 04_detect_image.py image.jpg(含人/车/动物等常见目标)")

    print(f"检测到 {len(results)} 个目标(score>0.5),结果已保存至 det_out.jpg")
    for r in results:
        print(f"   {r.label:<15} {r.score:.2f}  box=({r.box[0]:.0f},{r.box[1]:.0f},{r.box[2]:.0f},{r.box[3]:.0f})")
    print(f"\n   该模型可识别 {len(det.classes)} 类 COCO 目标。")
    print("   FPN 中含一层 DCN(identity 初始化),后续将其部署到 TRT 插件(混合精度)。")


if __name__ == "__main__":
    main()
