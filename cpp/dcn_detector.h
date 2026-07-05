// dcn_detector.h —— 基于 TrtEngine 实现 RetinaNet 后处理(anchor 解码 + class-aware NMS),输出检测框。
//
//   引擎(det.engine)输出每个 anchor 的 cls_logits[A, K] 与 bbox_reg[A, 4](检测头已在引擎内,由 GPU 计算);
//   本类仅负责引擎之外的纯 C++ 后处理(CPU 数组运算,非 CUDA kernel):
//     逐 FPN 层 阈值 + topk -> BoxCoder 解码 -> clip -> 跨层 class-aware NMS -> top-N。
//   与 python/08 的 numpy postprocess() 逐步对应,可对齐 det_ref.txt 进行数值验证。
//   anchors 由 Python(torchvision AnchorGenerator)预生成并存入 anchors.bin,此处直接读取,
//   以避免在 C++ 中重复实现易错的 anchor 生成逻辑。

#pragma once
#include "trt_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace dcn {

// 单条检测结果。
struct Det {
    int label;          // 类别索引(对应 categories())
    float score;        // 置信度
    float x1, y1, x2, y2;  // 框(输入图像像素坐标)
};

// 后处理各阶段耗时(ms,累计),供 bench() 定位 CPU 侧瓶颈所在阶段。
struct Prof {
    double d2h = 0;   // getOutput 将 logits/reg 由设备拷回主机(D2H)及 vector 分配
    double cls = 0;   // 逐 FPN 层的 sigmoid 阈值 + topk + 解码(遍历全部 anchor)
    double nms = 0;   // class-aware NMS(仅作用于候选框)
};

class DCNDetector {
public:
    // enginePath=det.engine;pluginPath=libdcnv2.so;auxDir=存放 anchors.bin / det_meta.txt / det_categories.txt 的目录。
    DCNDetector(const std::string& enginePath, const std::string& pluginPath,
                const std::string& auxDir = ".")
        : mEngine(enginePath, pluginPath) {
        loadMeta(auxDir + "/det_meta.txt");                 // 读取尺寸 / 类数 / 阈值 / 各层 anchor 数
        loadCategories(auxDir + "/det_categories.txt");     // 读取类名
        loadAnchors(auxDir + "/anchors.bin");               // 读取预生成的 anchors[A, 4]
    }

    int inputElems() const { return 3 * mSize * mSize; }    // 输入元素数(C, H, W)
    int inputSize() const { return mSize; }
    const std::vector<std::string>& categories() const { return mCats; }

    // 端到端检测:输入预处理后的图像(float CHW,[1, 3, size, size]),返回按分数降序的检测框。
    //   pf != nullptr 时,将各阶段耗时累加至 *pf(供 bench 分段统计;常规调用传 nullptr,计时开销可忽略)。
    std::vector<Det> detect(const float* input, Prof* pf = nullptr) {
        mEngine.setInput("input", input);                   // host -> device
        mEngine.infer();                                    // 引擎前向(backbone + DCN + head,GPU)
        auto _t0 = std::chrono::steady_clock::now();
        std::vector<float> cls = mEngine.getOutput("cls");  // [A*K] logits,由设备拷回主机(D2H,约 A*K*4 字节)
        std::vector<float> reg = mEngine.getOutput("reg");  // [A*4] 回归量(D2H)
        auto _t1 = std::chrono::steady_clock::now();

        // 在 logit 域进行阈值判定:sigmoid 单调递增,故 sigmoid(x) > thr 等价于 x > logit(thr)。
        // 内层循环仅比较原始 logit,避免对全部 anchor 求 sigmoid;仅对过阈值的候选计算概率,
        // 使 exp 调用量下降约 99%。
        const float logitThresh = std::log(mScoreThresh / (1.0f - mScoreThresh));  // 0.05 -> -2.944

        std::vector<Det> cand;                              // 跨层候选框
        int off = 0;                                        // 当前层的全局 anchor 起始下标
        for (int hwa : mHWA) {                              // 逐 FPN 层(顺序与 anchors / 引擎输出一致)
            // 1) 阈值:收集该层所有 logit > logitThresh 的 (分数, 全局 anchor, 类别)。
            std::vector<Cand> kept;
            for (int local = 0; local < hwa; ++local) {
                int ga = off + local;                       // 全局 anchor 下标
                for (int k = 0; k < mK; ++k) {
                    float logit = cls[(size_t)ga * mK + k];         // 原始 logit,暂不计算 sigmoid
                    if (logit > logitThresh)                        // 保序比较,等价于 sigmoid(logit) > thr
                        kept.push_back({sigmoidf(logit), ga, k});   // 仅对候选计算概率,结果与逐点 sigmoid 一致
                }
            }
            // 2) topk:按分数降序,每层最多保留 topk 个(与 python 一致:min(topk, 候选数))。
            std::stable_sort(kept.begin(), kept.end(),
                             [](const Cand& a, const Cand& b) { return a.s > b.s; });
            if ((int)kept.size() > mTopk) kept.resize(mTopk);
            // 3) 解码 + clip。
            for (const Cand& c : kept) {
                Det d = decode(&reg[(size_t)c.a * 4], &mAnchors[(size_t)c.a * 4]);
                d.label = c.l;
                d.score = c.s;
                cand.push_back(d);
            }
            off += hwa;
        }
        auto _t2 = std::chrono::steady_clock::now();

        // 4) 跨层 class-aware NMS 并限制总数。
        std::vector<Det> out = nmsClassAware(cand);
        auto _t3 = std::chrono::steady_clock::now();
        if (pf) {                                           // 仅 bench 分段轮传入非空 pf
            using dms = std::chrono::duration<double, std::milli>;
            pf->d2h += dms(_t1 - _t0).count();
            pf->cls += dms(_t2 - _t1).count();
            pf->nms += dms(_t3 - _t2).count();
        }
        return out;
    }

    // 延迟基准:分两层度量 —— "引擎算力"与"实际每帧延迟"是两个不同口径。
    //   ① GPU 纯推理:仅度量 enqueueV3 的 GPU 耗时(cudaEvent),不含 H2D / D2H / 后处理;
    //      为 INT8 与 FP16 的引擎算力对比口径(与 06 / dcn_infer 一致),更换引擎运行两次即可对比。
    //   ② 端到端 detect():以墙钟度量 setInput(H2D)+ infer + getOutput(D2H)+ 解码 + NMS,即每帧实际延迟。
    //   两者之差约等于 CPU 后处理 + 主机/设备拷贝;RetinaNet 全 anchor 输出时,该部分常显著大于 GPU 推理本身。
    // iters 为计时帧数;warmup 为预热帧数(先空跑若干帧,待惰性分配 / 算法选择 / GPU 时钟稳定,避免首帧偏慢污染统计)。
    void bench(const float* input, int iters = 200, int warmup = 20) {
        detect(input);                                      // 先完整运行一次:拷贝输入到设备并触发引擎惰性初始化
        float gpu_ms = mEngine.benchmark(iters);            // ① 纯 GPU:复用 TrtEngine 的 cudaEvent 计时(其内部自带一次预热)
        for (int i = 0; i < warmup; ++i) detect(input);     // ② 端到端预热(丢弃前 warmup 帧)
        std::vector<double> ts;                             // 记录每帧端到端耗时(ms),排序后取分位数
        ts.reserve(iters);
        for (int i = 0; i < iters; ++i) {                   // ② 逐帧墙钟计时(steady_clock 单调,适合度量时长)
            auto t0 = std::chrono::steady_clock::now();
            detect(input);                                  // 完整一帧:H2D + 推理 + D2H + 解码 + NMS
            auto t1 = std::chrono::steady_clock::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());  // 转换为毫秒
        }
        std::sort(ts.begin(), ts.end());                    // 升序排序,便于取 P50 / P90(分位数比均值更能反映抖动)
        double mean = 0.0;
        for (double v : ts) mean += v;
        mean /= ts.size();
        Prof pf;                                            // ③ 后处理分段:单独一轮,将 CPU 侧拆分为 D2H / 解码 / NMS
        for (int i = 0; i < iters; ++i) detect(input, &pf); // (与 ② 分开运行,避免污染端到端墙钟)
        // P50 为中位数(典型延迟);P90 为第 90 百分位(尾部延迟,反映偶发抖动)。
        std::printf("[bench] GPU 纯推理 enqueueV3(不含 H2D/D2H/后处理): %.3f ms/帧 (mean, %d 次)\n",
                    gpu_ms, iters);
        std::printf("[bench] 端到端 detect()(H2D+推理+D2H+解码+NMS): P50=%.3f  P90=%.3f  mean=%.3f ms (%d 次)\n",
                    ts[ts.size() / 2], ts[(size_t)(ts.size() * 0.9)], mean, iters);
        std::printf("[bench] 后处理分段(均值): D2H拷回=%.3f  逐层sigmoid阈值+解码=%.3f  NMS=%.3f ms\n",
                    pf.d2h / iters, pf.cls / iters, pf.nms / iters);
    }

private:
    struct Cand { float s; int a; int l; };   // 候选:分数 / 全局 anchor / 类别

    static float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    // anchor + 回归量 -> 框(torchvision BoxCoder,weights = 1),并 clip 到 [0, size]。
    Det decode(const float* reg, const float* anc) const {
        float w = anc[2] - anc[0], h = anc[3] - anc[1];
        float cx = anc[0] + 0.5f * w, cy = anc[1] + 0.5f * h;
        float dx = reg[0], dy = reg[1];
        float dw = std::fmin(reg[2], mClip), dh = std::fmin(reg[3], mClip);   // 限幅,防止 exp 溢出
        float pcx = dx * w + cx, pcy = dy * h + cy;     // 预测框中心
        float pw = std::exp(dw) * w, ph = std::exp(dh) * h;   // 预测框宽高
        Det d;
        d.x1 = clip(pcx - 0.5f * pw); d.y1 = clip(pcy - 0.5f * ph);
        d.x2 = clip(pcx + 0.5f * pw); d.y2 = clip(pcy + 0.5f * ph);
        return d;
    }
    float clip(float v) const { return v < 0.f ? 0.f : (v > (float)mSize ? (float)mSize : v); }

    static float iou(const Det& a, const Det& b) {
        float x1 = std::fmax(a.x1, b.x1), y1 = std::fmax(a.y1, b.y1);
        float x2 = std::fmin(a.x2, b.x2), y2 = std::fmin(a.y2, b.y2);
        float inter = std::fmax(0.f, x2 - x1) * std::fmax(0.f, y2 - y1);
        float aa = (a.x2 - a.x1) * (a.y2 - a.y1), ab = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (aa + ab - inter + 1e-9f);
    }

    // class-aware 贪心 NMS:按分数降序,仅抑制同类且 IoU > thr 的后续框;最终截断至 det_per_img。
    std::vector<Det> nmsClassAware(std::vector<Det>& c) const {
        std::stable_sort(c.begin(), c.end(),
                         [](const Det& a, const Det& b) { return a.score > b.score; });
        std::vector<char> dead(c.size(), 0);
        std::vector<Det> out;
        for (size_t i = 0; i < c.size(); ++i) {
            if (dead[i]) continue;
            out.push_back(c[i]);
            if ((int)out.size() >= mDetPerImg) break;       // 限制总数
            for (size_t j = i + 1; j < c.size(); ++j)
                if (!dead[j] && c[i].label == c[j].label && iou(c[i], c[j]) > mNmsThresh)
                    dead[j] = 1;                            // 同类且高重叠,予以抑制
        }
        return out;
    }

    // ---- 辅助文件加载 ----
    void loadMeta(const std::string& p) {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("无法打开 " + p + "(请先运行 08_export_det_engine.py)");
        std::string key, line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            ss >> key;
            if (key == "size") ss >> mSize;
            else if (key == "num_classes") ss >> mK;
            else if (key == "num_anchors") ss >> mA;
            else if (key == "score_thresh") ss >> mScoreThresh;
            else if (key == "nms_thresh") ss >> mNmsThresh;
            else if (key == "topk") ss >> mTopk;
            else if (key == "det_per_img") ss >> mDetPerImg;
            else if (key == "hwa") { int v; while (ss >> v) mHWA.push_back(v); }
        }
        mClip = std::log(1000.0f / 16.0f);                  // 与 python 的 CLIP 保持一致
    }
    void loadCategories(const std::string& p) {
        std::ifstream f(p);
        std::string line;
        while (std::getline(f, line)) mCats.push_back(line);
    }
    void loadAnchors(const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) throw std::runtime_error("无法打开 " + p);
        mAnchors.resize((size_t)mA * 4);
        f.read(reinterpret_cast<char*>(mAnchors.data()), mAnchors.size() * sizeof(float));
    }

    TrtEngine mEngine;                 // 底层引擎
    std::vector<float> mAnchors;       // [A, 4] 预生成 anchors
    std::vector<std::string> mCats;    // 类名
    std::vector<int> mHWA;             // 各 FPN 层的 anchor 数
    int mSize = 512, mK = 0, mA = 0, mTopk = 1000, mDetPerImg = 300;
    float mScoreThresh = 0.05f, mNmsThresh = 0.5f, mClip = 4.135f;
};

}  // namespace dcn
