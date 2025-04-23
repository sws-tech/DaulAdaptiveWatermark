#include "RegionScorer.h"
#include "utils.h" // 需要 calculateEntropy
#include <numeric> // for std::accumulate

RegionScorer::RegionScorer(double alpha, double beta, double gamma, double delta)
    : weightAlpha(alpha), weightBeta(beta), weightGamma(gamma), weightDelta(delta) {}

// 计算区域的所有得分并更新 Region 对象
void RegionScorer::calculateRegionScores(Region& region, const cv::Mat& originalPatch, const cv::Mat& edgePatch, const cv::Point& imageCenter) {
    if (originalPatch.empty() || edgePatch.empty() || originalPatch.size() != edgePatch.size()) {
         throw std::runtime_error("RegionScorer: Input patches are invalid or mismatched.");
    }
    if (originalPatch.channels() != 1 || edgePatch.channels() != 1) {
        throw std::runtime_error("RegionScorer: Patches must be single-channel grayscale.");
    }

    region.edgeScore = calculateEdgeScore(edgePatch);
    region.textureScore = calculateTextureScore(originalPatch);
    region.grayScore = calculateGrayScore(originalPatch);
    region.positionScore = calculatePositionScore(region.center, imageCenter, region.bounds.width, region.bounds.height);

    region.score = calculateCombinedScore(region.edgeScore, region.textureScore, region.grayScore, region.positionScore);
}


// 计算边缘得分 E_uv (式 2)
double RegionScorer::calculateEdgeScore(const cv::Mat& edgePatch) {
    int m = edgePatch.rows;
    int n = edgePatch.cols;
    if (m == 0 || n == 0) return 0.0;

    // 计算边缘像素数量 (值为 255)
    int edgePixelCount = cv::countNonZero(edgePatch); // OpenCV 函数直接计数非零元素

    // 加上 1 避免除以零
    double denominator = static_cast<double>(edgePixelCount) + 1.0;

    // 计算得分
    double score = std::sqrt(static_cast<double>(m * n) / denominator);
    return score;
}

// 计算纹理得分 H_uv (式 3) - 使用信息熵
double RegionScorer::calculateTextureScore(const cv::Mat& originalPatch) {
    return calculateEntropy(originalPatch); // 直接调用辅助函数
}

// 计算灰度得分 G_uv (式 4)
double RegionScorer::calculateGrayScore(const cv::Mat& originalPatch) {
    int m = originalPatch.rows;
    int n = originalPatch.cols;
    if (m == 0 || n == 0) return 0.0;

    double sumAbsDiff = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            sumAbsDiff += std::abs(128.0 - static_cast<double>(originalPatch.at<uchar>(i, j)));
        }
    }

    double avgAbsDiff = sumAbsDiff / (m * n);

    // 加上一个小的 epsilon 防止 log2(0)
    double epsilon = 1e-9;
    double score = std::abs(std::log2(avgAbsDiff + epsilon)); // 使用 log base 2
    return score;
}

// 计算位置得分 P_uv (式 5)
double RegionScorer::calculatePositionScore(const cv::Point& regionCenter, const cv::Point& imageCenter, int regionWidth, int regionHeight) {
    if (regionWidth == 0 || regionHeight == 0) return 0.0;

    double distSq = std::pow(regionCenter.x - imageCenter.x, 2) + std::pow(regionCenter.y - imageCenter.y, 2);

    // 加上 0.01 避免除以零
    double denominator = distSq + 0.01;

    double score = std::sqrt(static_cast<double>(2 * regionWidth * regionHeight) / denominator);
    return score;
}

// 计算综合得分 Score_uv (式 6)
double RegionScorer::calculateCombinedScore(double E, double H, double G, double P) {
    return weightAlpha * E + weightBeta * H + weightGamma * G + weightDelta * P;
}
