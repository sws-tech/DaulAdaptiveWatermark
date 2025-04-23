#ifndef WATERMARK_EMBEDDER_H
#define WATERMARK_EMBEDDER_H

#include "EdgeDetector.h"
#include "RegionSelector.h"
#include "WatermarkEncoder.h"
#include "BlockProcessor.h"
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class WatermarkEmbedder {
public:
    // 构造函数，初始化各个组件
    WatermarkEmbedder(int numRegions = 10, int edgeThreshold = 3); // 示例参数

    // 执行完整的水印嵌入过程
    cv::Mat embedWatermark(const cv::Mat& originalImage, const std::string& watermarkText);

private:
    EdgeDetector edgeDetector;
    RegionScorer regionScorer; // RegionSelector 内部会用到
    RegionSelector regionSelector;
    WatermarkEncoder watermarkEncoder;
    BlockProcessor blockProcessor;

    int numberOfRegions; // d
};

#endif // WATERMARK_EMBEDDER_H
