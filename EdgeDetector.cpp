#include "EdgeDetector.h"
#include "utils.h" // 需要 DCT/IDCT

EdgeDetector::EdgeDetector(double lowThresh, double highThresh, double postProcessThresh)
    : cannyLowThreshold(lowThresh), cannyHighThreshold(highThresh), postProcessingThreshold(postProcessThresh) {}

cv::Mat EdgeDetector::detectEdges(const cv::Mat& originalImage) {
    if (originalImage.empty() || originalImage.channels() != 1) {
        throw std::runtime_error("EdgeDetector: Input image must be a single-channel grayscale image.");
    }

    // Step 1.1: 预处理
    cv::Mat preprocessedImage = preProcess(originalImage);

    // Step 1.2: Canny 边缘检测
    cv::Mat cannyEdges;
    cv::Canny(preprocessedImage, cannyEdges, cannyLowThreshold, cannyHighThreshold);

    // Step 1.3: 后处理
    cv::Mat finalEdges = postProcess(cannyEdges, originalImage); // 注意：后处理使用原始图像计算灰度差

    return finalEdges;
}

// 预处理：DCT域噪声抑制
cv::Mat EdgeDetector::preProcess(const cv::Mat& image) {
    // 修改为CV_64F精度
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_64F);

    // 计算 DCT
    cv::Mat dctCoeffs = calculateDCT(floatImage);    // --- 自适应DCT系数保留策略（优化版本）---
    int rows = dctCoeffs.rows;
    int cols = dctCoeffs.cols;
    std::vector<std::pair<double*, double>> acCoeffData; // (指针, 绝对值)

    // 计算AC系数的方差以确定自适应阈值
    double mean = 0.0, variance = 0.0;
    int acCount = 0;

    int r = 0, c = 0;
    bool up = true;
    for (int i = 1; i < rows * cols; ++i) {
        if (r != 0 || c != 0) {
             if (dctCoeffs.at<double>(r, c) != 0.0) {
                 double absValue = std::abs(dctCoeffs.at<double>(r, c));
                 acCoeffData.push_back({&dctCoeffs.at<double>(r, c), absValue});
                 mean += absValue;
                 acCount++;
             }
        }
        if (up) {
            if (r > 0 && c < cols - 1) { r--; c++; }
            else {
                up = false;
                if (c < cols - 1) c++; else r++;
            }
        } else {
            if (c > 0 && r < rows - 1) { c--; r++; }
            else {
                up = true;
                if (r < rows - 1) r++; else c++;
            }
        }
        if (r >= rows || c >= cols) break;
    }

    if (acCount > 0) {
        mean /= acCount;
        
        // 计算方差
        for (const auto& coeffData : acCoeffData) {
            variance += (coeffData.second - mean) * (coeffData.second - mean);
        }
        variance /= acCount;
        
        // 自适应阈值：基于方差的动态保留策略
        double retentionRatio;
        if (variance > mean * mean) {
            // 高方差区域：更多细节，保留更多系数（保留15-5%，即置零85-95%）
            retentionRatio = 0.15 - 0.10 * std::min(1.0, variance / (mean * mean * 4));
        } else {
            // 低方差区域：平滑区域，可以更大胆去噪（保留5-10%，即置零90-95%）
            retentionRatio = 0.05 + 0.05 * (variance / (mean * mean));
        }
        
        // 按绝对值排序，保留幅值较大的系数（优先保留重要系数）
        std::sort(acCoeffData.begin(), acCoeffData.end(), 
                 [](const std::pair<double*, double>& a, const std::pair<double*, double>& b) {
                     return a.second > b.second;
                 });
        
        // 置零较小的系数
        int numToZero = static_cast<int>(acCoeffData.size() * (1.0 - retentionRatio));
        for (int i = acCoeffData.size() - numToZero; i < acCoeffData.size(); ++i) {
            *(acCoeffData[i].first) = 0.0;
        }
    }
    // --- 自适应DCT系数处理结束 ---

    // 计算 IDCT
    cv::Mat idctResult = calculateIDCT(dctCoeffs);

    // 转换回 8位无符号整数类型，并进行截断
    cv::Mat processedImage;
    idctResult.convertTo(processedImage, CV_8U);
    cv::normalize(processedImage, processedImage, 0, 255, cv::NORM_MINMAX);

    return processedImage;
}

// 后处理：去除误检边缘 (式 1)
cv::Mat EdgeDetector::postProcess(const cv::Mat& edgeImage, const cv::Mat& originalImage) {
    cv::Mat processedEdges = edgeImage.clone(); // 创建副本进行修改

    for (int r = 1; r < edgeImage.rows - 1; ++r) {
        for (int c = 1; c < edgeImage.cols - 1; ++c) {
            // 只处理 Canny 检测到的边缘像素 (值为 255)
            if (edgeImage.at<uchar>(r, c) == 255) {
                double sumDiff = 0.0;
                uchar centerPixelValue = originalImage.at<uchar>(r, c);

                // 计算与八邻域的灰度差绝对值之和
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (dr == 0 && dc == 0) continue; // 跳过中心像素
                        sumDiff += std::abs(static_cast<int>(originalImage.at<uchar>(r + dr, c + dc)) - static_cast<int>(centerPixelValue));
                    }
                }

                // 计算平均灰度差 A_diff
                double avgDiff = sumDiff / 8.0;

                // 如果平均差小于阈值 t，则认为是误检，置为 0
                if (avgDiff < postProcessingThreshold) {
                    processedEdges.at<uchar>(r, c) = 0;
                }
            }
        }
    }
    return processedEdges;
}
