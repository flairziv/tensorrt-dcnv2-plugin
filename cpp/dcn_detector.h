// dcn_detector.h —— 在 TrtEngine 之上实现 RetinaNet 后处理(anchor 解码 + class-aware NMS),输出检测框。
//
//   引擎(det.engine)输出每个 anchor 的 cls_logits[A,K] 与 bbox_reg[A,4](检测头已在引擎内,GPU 计算);
//   本类仅完成引擎外的纯 C++ 后处理(CPU 数组运算,非 CUDA kernel):
//     逐 FPN 层 阈值 + topk -> BoxCoder 解码 -> clip -> 跨层 class-aware 贪心 NMS -> top-N。
//   与 python/08 的 numpy postprocess() 一一对应,可逐行对齐 det_ref.txt 进行验证。
//   anchors 由 Python(torchvision AnchorGenerator)预生成并存入 anchors.bin,此处直接读取,
//   避免在 C++ 中重写易错的 anchor 生成逻辑。

#pragma once
#include "trt_engine.h"

#include <algorithm>
#include <cmath>
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

class DCNDetector {
public:
    // enginePath=det.engine;pluginPath=libdcnv2.so;auxDir=存放 anchors.bin / det_meta.txt / det_categories.txt 的目录。
    DCNDetector(const std::string& enginePath, const std::string& pluginPath,
                const std::string& auxDir = ".")
        : mEngine(enginePath, pluginPath) {
        loadMeta(auxDir + "/det_meta.txt");                 // 读取尺寸/类数/阈值/各层 anchor 数
        loadCategories(auxDir + "/det_categories.txt");     // 读取类名
        loadAnchors(auxDir + "/anchors.bin");               // 读取预生成 anchors[A,4]
    }

    int inputElems() const { return 3 * mSize * mSize; }    // 输入元素数(C,H,W)
    int inputSize() const { return mSize; }
    const std::vector<std::string>& categories() const { return mCats; }

    // 端到端检测:输入预处理后的图像(float CHW,[1,3,size,size]),返回按分数降序的检测框。
    std::vector<Det> detect(const float* input) {
        mEngine.setInput("input", input);                   // host -> device
        mEngine.infer();                                    // 引擎前向(backbone + DCN + head,GPU)
        std::vector<float> cls = mEngine.getOutput("cls");  // [A*K] logits(行主序,anchor 优先)
        std::vector<float> reg = mEngine.getOutput("reg");  // [A*4] 回归量

        std::vector<Det> cand;                              // 跨层候选
        int off = 0;                                        // 当前层的全局 anchor 起始
        for (int hwa : mHWA) {                              // 逐 FPN 层(顺序与 anchors / 引擎输出一致)
            // 1) 阈值:收集该层所有 score > thr 的 (分数, 全局 anchor, 类别)。
            std::vector<Cand> kept;
            for (int local = 0; local < hwa; ++local) {
                int ga = off + local;                       // 全局 anchor 下标
                for (int k = 0; k < mK; ++k) {
                    float s = sigmoidf(cls[(size_t)ga * mK + k]);   // logit -> 概率
                    if (s > mScoreThresh) kept.push_back({s, ga, k});
                }
            }
            // 2) topk:按分数降序,每层最多保留 topk 个。
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

        // 4) 跨层 class-aware NMS 并限制总数。
        return nmsClassAware(cand);
    }

private:
    struct Cand { float s; int a; int l; };   // 候选:分数 / 全局 anchor / 类别

    static float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    // anchor + 回归量 -> 框(torchvision BoxCoder,weights=1),并 clip 到 [0,size]。
    Det decode(const float* reg, const float* anc) const {
        float w = anc[2] - anc[0], h = anc[3] - anc[1];
        float cx = anc[0] + 0.5f * w, cy = anc[1] + 0.5f * h;
        float dx = reg[0], dy = reg[1];
        float dw = std::fmin(reg[2], mClip), dh = std::fmin(reg[3], mClip);   // 限幅防 exp 溢出
        float pcx = dx * w + cx, pcy = dy * h + cy;     // 预测中心
        float pw = std::exp(dw) * w, ph = std::exp(dh) * h;   // 预测宽高
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

    // class-aware 贪心 NMS:分数降序,仅抑制同类且 IoU > thr 的后续框;最终截断到 det_per_img。
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
                    dead[j] = 1;                            // 同类高重叠则抑制
        }
        return out;
    }

    // 加载辅助文件。
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
        mClip = std::log(1000.0f / 16.0f);                  // 与 python CLIP 一致
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
    std::vector<float> mAnchors;       // [A,4] 预生成 anchors
    std::vector<std::string> mCats;    // 类名
    std::vector<int> mHWA;             // 各 FPN 层的 anchor 数
    int mSize = 512, mK = 0, mA = 0, mTopk = 1000, mDetPerImg = 300;
    float mScoreThresh = 0.05f, mNmsThresh = 0.5f, mClip = 4.135f;
};

}  // namespace dcn
