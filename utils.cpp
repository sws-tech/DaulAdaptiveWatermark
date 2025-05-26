#include "utils.h"
#include <map>
#include <numeric>

// 计算DCT（使用OpenCV）
cv::Mat calculateDCT(const cv::Mat& input) {
    cv::Mat floatInput;
    // 确保输入是浮点类型
    if (input.type() != CV_32F && input.type() != CV_64F) {
        input.convertTo(floatInput, CV_32F);
    } else {
        floatInput = input.clone();
    }
    cv::Mat dctResult;
    cv::dct(floatInput, dctResult);
    return dctResult;
}

// 计算IDCT（使用OpenCV）
cv::Mat calculateIDCT(const cv::Mat& input) {
    cv::Mat idctResult;
    cv::idct(input, idctResult);
    return idctResult;
}

// 计算信息熵 (式 3 的核心部分)
double calculateEntropy(const cv::Mat& region) {
    std::map<uchar, int> hist;
    int totalPixels = region.rows * region.cols;
    if (totalPixels == 0) return 0.0;

    for (int i = 0; i < region.rows; ++i) {
        for (int j = 0; j < region.cols; ++j) {
            hist[region.at<uchar>(i, j)]++;
        }
    }

    double entropy = 0.0;
    for (auto const& [pixelValue, count] : hist) {
        if (count > 0) {
            double probability = static_cast<double>(count) / totalPixels;
            entropy -= probability * std::log2(probability); // 使用 log base 2
        }
    }
    return entropy; // 这就是 H_uv
}

// 计算高斯权重 (式 17) - 用于边缘块像素修改量分配
cv::Mat calculateGaussianWeights(int rows, int cols, double sigma) {
    cv::Mat weights = cv::Mat::zeros(rows, cols, CV_64F);
    double sumWeights = 0.0;
    int centerX = cols / 2;
    int centerY = rows / 2;
    double sigmaSq = sigma * sigma;
    double factor = 1.0 / (2.0 * CV_PI * sigmaSq);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double distSq = std::pow(i - centerY, 2) + std::pow(j - centerX, 2);
            double weight = factor * std::exp(-distSq / (2.0 * sigmaSq));
            weights.at<double>(i, j) = weight;
            sumWeights += weight;
        }
    }

    // 归一化权重，使其和为 1
    if (sumWeights > 1e-9) { // 避免除以零
        weights /= sumWeights;
    }
    return weights;
}
