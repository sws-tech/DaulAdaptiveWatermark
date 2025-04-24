#ifndef REGION_SELECTOR_H
#define REGION_SELECTOR_H

#include "utils.h"
#include "RegionScorer.h"
#include <vector>
#include <opencv2/opencv.hpp>

class RegionSelector {
public:
    // 构造函数，传入评分器、目标区域数量 d、滑窗比例 a、步长比例 b
    RegionSelector(RegionScorer scorer, int numRegionsToSelect = 10, double windowScale = 0.25, double stepScale = 0.25);

    // 选择嵌入区域 (对应 Step 2 主要逻辑)
    std::vector<Region> selectEmbeddingRegions(const cv::Mat& originalImage, const cv::Mat& edgeImage);

    // 新增 Getter 方法
    double getWindowScale() const { return windowSizeScale; }
    double getStepScale() const { return stepSizeScale; }

private:
    RegionScorer regionScorer;
    int targetRegionCount; // d
    double windowSizeScale; // a: 滑窗大小与图像尺寸的比例
    double stepSizeScale;   // b: 步长与窗口大小的比例
};

#endif // REGION_SELECTOR_H
