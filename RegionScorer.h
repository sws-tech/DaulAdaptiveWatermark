#ifndef REGION_SCORER_H
#define REGION_SCORER_H

#include "utils.h"
#include <opencv2/opencv.hpp>

class RegionScorer {
public:
    // 构造函数，可以传入权重 alpha, beta, gamma, delta
    RegionScorer(double alpha = 0.4, double beta = 0.2, double gamma = 0.2, double delta = 0.2);

    // 计算单个区域的综合得分 (对应 Step 2 部分计算)
    void calculateRegionScores(Region& region, const cv::Mat& originalPatch, const cv::Mat& edgePatch, const cv::Point& imageCenter);

    // 计算边缘得分 E_uv (式 2)
    double calculateEdgeScore(const cv::Mat& edgePatch);

    // 计算纹理得分 H_uv (式 3) - 使用信息熵
    double calculateTextureScore(const cv::Mat& originalPatch);

    // 计算灰度得分 G_uv (式 4)
    double calculateGrayScore(const cv::Mat& originalPatch);

    // 计算位置得分 P_uv (式 5)
    double calculatePositionScore(const cv::Point& regionCenter, const cv::Point& imageCenter, int regionWidth, int regionHeight);

    // 计算综合得分 Score_uv (式 6)
    double calculateCombinedScore(double E, double H, double G, double P);

private:
    double weightAlpha; // alpha
    double weightBeta;  // beta
    double weightGamma; // gamma
    double weightDelta; // delta
};

#endif // REGION_SCORER_H
