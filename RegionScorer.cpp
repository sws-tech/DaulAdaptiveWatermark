#include "RegionScorer.h"
#include "utils.h" // ��Ҫ calculateEntropy
#include <numeric> // for std::accumulate

RegionScorer::RegionScorer(double alpha, double beta, double gamma, double delta)
    : weightAlpha(alpha), weightBeta(beta), weightGamma(gamma), weightDelta(delta) {}

// ������������е÷ֲ����� Region ����
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


// �����Ե�÷� E_uv (ʽ 2)
double RegionScorer::calculateEdgeScore(const cv::Mat& edgePatch) {
    int m = edgePatch.rows;
    int n = edgePatch.cols;
    if (m == 0 || n == 0) return 0.0;

    // �����Ե�������� (ֵΪ 255)
    int edgePixelCount = cv::countNonZero(edgePatch); // OpenCV ����ֱ�Ӽ�������Ԫ��

    // ���� 1 ���������
    double denominator = static_cast<double>(edgePixelCount) + 1.0;

    // ����÷�
    double score = std::sqrt(static_cast<double>(m * n) / denominator);
    return score;
}

// ��������÷� H_uv (ʽ 3) - ʹ����Ϣ��
double RegionScorer::calculateTextureScore(const cv::Mat& originalPatch) {
    return calculateEntropy(originalPatch); // ֱ�ӵ��ø�������
}

// ����Ҷȵ÷� G_uv (ʽ 4)
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

    // ����һ��С�� epsilon ��ֹ log2(0)
    double epsilon = 1e-9;
    double score = std::abs(std::log2(avgAbsDiff + epsilon)); // ʹ�� log base 2
    return score;
}

// ����λ�õ÷� P_uv (ʽ 5)
double RegionScorer::calculatePositionScore(const cv::Point& regionCenter, const cv::Point& imageCenter, int regionWidth, int regionHeight) {
    if (regionWidth == 0 || regionHeight == 0) return 0.0;

    double distSq = std::pow(regionCenter.x - imageCenter.x, 2) + std::pow(regionCenter.y - imageCenter.y, 2);

    // ���� 0.01 ���������
    double denominator = distSq + 0.01;

    double score = std::sqrt(static_cast<double>(2 * regionWidth * regionHeight) / denominator);
    return score;
}

// �����ۺϵ÷� Score_uv (ʽ 6)
double RegionScorer::calculateCombinedScore(double E, double H, double G, double P) {
    return weightAlpha * E + weightBeta * H + weightGamma * G + weightDelta * P;
}
