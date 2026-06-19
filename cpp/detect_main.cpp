// detect_main.cpp —— 纯 C++ 端到端检测示例,两种模式:
//   1. 给定图片路径(需编译期找到 OpenCV):imread -> 预处理 -> DCNDetector.detect() -> 画框存 det_out.jpg。
//   2. 不给图片:读取 det_input.bin -> detect -> 与 Python 参考 det_ref.txt 逐行对齐。
//
// 运行(在 python/ 目录,辅助文件位于此处):
//   ../cpp/build/detect det.engine ../src/build/libdcnv2.so .                 # bin 对齐模式
//   ../cpp/build/detect det.engine ../src/build/libdcnv2.so . image.jpg       # 图片模式(需 OpenCV)

#include "dcn_detector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>

// 预处理:BGR 图 -> resize 到 size*size -> RGB -> [0,1] -> ImageNet 归一化 -> CHW float(与 python/08 的 make_input 一致)。
//   引擎为去除 transform 的 HeadForward,须输入已归一化张量;布局为 CHW(单 batch 的 NCHW)。
static std::vector<float> preprocess(const cv::Mat& bgr, int size) {
    cv::Mat resized, rgb;
    cv::resize(bgr, resized, cv::Size(size, size));        // 缩放至固定尺寸(OpenCV 默认双线性,与 torch antialias 略有差异)
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);         // BGR -> RGB(OpenCV 读入为 BGR,模型需 RGB)
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);               // uint8 -> float[0,1]
    const float mean[3] = {0.485f, 0.456f, 0.406f};        // ImageNet 均值
    const float stdv[3] = {0.229f, 0.224f, 0.225f};        // ImageNet 标准差
    std::vector<float> out((size_t)3 * size * size);
    for (int c = 0; c < 3; ++c)                            // HWC -> CHW,并逐通道减均值除标准差
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
                out[((size_t)c * size + y) * size + x] =
                    (rgb.at<cv::Vec3f>(y, x)[c] - mean[c]) / stdv[c];
    return out;
}

// 图片模式:读入一张图 -> 检测 -> 画框存盘。
static int runImage(dcn::DCNDetector& det, const std::string& imgPath) {
    cv::Mat bgr = cv::imread(imgPath);                     // 读图(BGR,uint8)
    if (bgr.empty()) { fprintf(stderr, "无法读取图片 %s\n", imgPath.c_str()); return 1; }
    int S = det.inputSize();                               // 引擎固定输入边长(512)
    std::vector<float> input = preprocess(bgr, S);         // 预处理为引擎输入张量
    std::vector<dcn::Det> dets = det.detect(input.data()); // 端到端检测(坐标位于 S*S 空间)

    // 框坐标由 S*S 缩回原图尺寸(预处理为拉伸 resize,按比例缩回)。
    float sx = (float)bgr.cols / S, sy = (float)bgr.rows / S;
    const auto& cats = det.categories();
    const float SHOW = 0.3f;                               // 仅绘制 score>=0.3 的框(0.05 阈值的低分框过多)
    int drawn = 0;
    printf("[C++] 检测到 %zu 个框(score>=%.2f 的绘制到 det_out.jpg):\n", dets.size(), SHOW);
    for (const auto& d : dets) {
        if (d.score < SHOW) continue;
        const char* nm = (d.label >= 0 && d.label < (int)cats.size()) ? cats[d.label].c_str() : "?";
        int x1 = (int)(d.x1 * sx), y1 = (int)(d.y1 * sy), x2 = (int)(d.x2 * sx), y2 = (int)(d.y2 * sy);
        cv::rectangle(bgr, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);
        char tag[128]; snprintf(tag, sizeof(tag), "%s %.2f", nm, d.score);
        cv::putText(bgr, tag, {x1, std::max(0, y1 - 5)}, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, {0, 255, 0}, 1);
        printf("   %-14s %.3f  [%d,%d,%d,%d]\n", nm, d.score, x1, y1, x2, y2);
        ++drawn;
    }
    cv::imwrite("det_out.jpg", bgr);                       // 保存结果图
    printf("已绘制 %d 个框 -> det_out.jpg(纯 C++ 端到端:imread -> 预处理 -> 引擎 -> 解码+NMS -> 画框)\n", drawn);
    return 0;
}
#endif  // HAVE_OPENCV

// bin 对齐模式:读取 det_input.bin -> detect -> 与 Python 参考逐行对齐。
static int runBinParity(dcn::DCNDetector& det, const std::string& auxDir) {
    std::vector<float> input(det.inputElems());
    std::ifstream fin(auxDir + "/det_input.bin", std::ios::binary);
    if (!fin) { fprintf(stderr, "缺少 det_input.bin,请先运行 08_export_det_engine.py\n"); return 1; }
    fin.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));

    std::vector<dcn::Det> dets = det.detect(input.data());
    printf("[C++] 检测到 %zu 个框(分数降序,前 15):\n", dets.size());
    const auto& cats = det.categories();
    for (size_t i = 0; i < dets.size() && i < 15; ++i) {
        const auto& d = dets[i];
        const char* nm = (d.label >= 0 && d.label < (int)cats.size()) ? cats[d.label].c_str() : "?";
        printf("   %-14s %.3f  [%.1f, %.1f, %.1f, %.1f]\n", nm, d.score, d.x1, d.y1, d.x2, d.y2);
    }

    std::ifstream fref(auxDir + "/det_ref.txt");
    if (!fref) { printf("\n(无 det_ref.txt,跳过对齐)\n"); return 0; }
    int N; fref >> N;
    int nLabelMismatch = 0;
    double maxScoreDiff = 0, maxBoxDiff = 0;
    int cmp = (int)std::min((size_t)N, dets.size());
    for (int i = 0; i < N; ++i) {
        int lab; float sc, x1, y1, x2, y2;
        fref >> lab >> sc >> x1 >> y1 >> x2 >> y2;
        if (i >= (int)dets.size()) continue;
        const auto& d = dets[i];
        if (d.label != lab) nLabelMismatch++;
        maxScoreDiff = std::fmax(maxScoreDiff, std::fabs(sc - d.score));
        maxBoxDiff = std::fmax(maxBoxDiff,
            std::fmax(std::fmax(std::fabs(x1 - d.x1), std::fabs(y1 - d.y1)),
                      std::fmax(std::fabs(x2 - d.x2), std::fabs(y2 - d.y2))));
    }
    printf("\n[对齐] Python 参考 %d 个 / C++ %zu 个;前 %d 个:类别不一致 %d 处,"
           "max|score 差|=%.3e,max|box 差|=%.3e\n",
           N, dets.size(), cmp, nLabelMismatch, maxScoreDiff, maxBoxDiff);
    if (N == 0 && dets.empty()) {
        printf("双方均为 0 框,通常为随机输入所致。请传入真实图片重跑:\n"
               "   cd ../python && python 08_export_det_engine.py image.jpg\n");
    } else {
        bool ok = ((int)dets.size() == N) && nLabelMismatch == 0 && maxBoxDiff < 1.0;
        printf("%s\n", ok
            ? "[PASS] C++ 后处理与 Python 对齐(框与类别一致),纯 C++ 端到端检测验证通过"
            : "[FAIL] 存在偏差:检查逐层 topk 顺序 / 解码公式 / NMS 阈值 / anchors.bin 是否为同一次导出");
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string engPath = argc > 1 ? argv[1] : "det.engine";
    std::string pluginPath = argc > 2 ? argv[2] : "../src/build/libdcnv2.so";
    std::string auxDir = argc > 3 ? argv[3] : ".";          // anchors/meta/categories/输入/参考 所在目录
    std::string imgPath = argc > 4 ? argv[4] : "";          // 第 4 参数:图片路径(可选)

    try {
        dcn::DCNDetector det(engPath, pluginPath, auxDir);  // 加载引擎与辅助文件

        if (!imgPath.empty()) {                             // 提供图片则进入图片模式
#ifdef HAVE_OPENCV
            return runImage(det, imgPath);
#else
            fprintf(stderr, "本 detect 未链接 OpenCV,无法直接读取图片。\n"
                            "安装 OpenCV 后重新 cmake(自动检测)即可;或使用 bin 模式(不传图片参数)。\n");
            return 1;
#endif
        }
        return runBinParity(det, auxDir);                   // 未提供图片则进入 bin 对齐模式
    } catch (const std::exception& e) {
        fprintf(stderr, "错误: %s\n", e.what());
        return 1;
    }
}
