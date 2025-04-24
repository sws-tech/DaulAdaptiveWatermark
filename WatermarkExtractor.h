#ifndef WATERMARK_EXTRACTOR_H
#define WATERMARK_EXTRACTOR_H

#include "EdgeDetector.h"
#include "RegionSelector.h"
#include "BlockProcessor.h"
#include "WatermarkDecoder.h" // 新增解码器依赖
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class WatermarkExtractor {
public:
    // 构造函数，不再需要原始图像路径
    WatermarkExtractor(int expectedWatermarkLength, int edgeThreshold = 25);

    // 执行完整的水印提取过程
    std::string extractWatermark(const cv::Mat& watermarkedImage);

private:
    EdgeDetector edgeDetector;
    RegionScorer regionScorer; // RegionSelector 内部会用到
    RegionSelector regionSelector;
    BlockProcessor blockProcessor;
    WatermarkDecoder watermarkDecoder; // 新增解码器实例

    int expectedWatermarkLength; // m
};

#endif // WATERMARK_EXTRACTOR_H
