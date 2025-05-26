#include "utils.h"
#include <map>
#include <numeric>

// ����DCT��ʹ��OpenCV��
cv::Mat calculateDCT(const cv::Mat& input) {
    cv::Mat floatInput;
    // ȷ�������Ǹ�������
    if (input.type() != CV_32F && input.type() != CV_64F) {
        input.convertTo(floatInput, CV_32F);
    } else {
        floatInput = input.clone();
    }
    cv::Mat dctResult;
    cv::dct(floatInput, dctResult);
    return dctResult;
}

// ����IDCT��ʹ��OpenCV��
cv::Mat calculateIDCT(const cv::Mat& input) {
    cv::Mat idctResult;
    cv::idct(input, idctResult);
    return idctResult;
}

// ������Ϣ�� (ʽ 3 �ĺ��Ĳ���)
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
            entropy -= probability * std::log2(probability); // ʹ�� log base 2
        }
    }
    return entropy; // ����� H_uv
}

// �����˹Ȩ�� (ʽ 17) - ���ڱ�Ե�������޸�������
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

    // ��һ��Ȩ�أ�ʹ���Ϊ 1
    if (sumWeights > 1e-9) { // ���������
        weights /= sumWeights;
    }
    return weights;
}
