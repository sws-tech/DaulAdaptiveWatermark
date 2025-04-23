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

    cv::imwrite("canny_process.png", cannyEdges);

    // Step 1.3: 后处理
    cv::Mat finalEdges = postProcess(cannyEdges, originalImage); // 注意：后处理使用原始图像计算灰度差

    cv::imwrite("post_process.png", finalEdges);
    return finalEdges;
}

// 预处理：DCT域噪声抑制
cv::Mat EdgeDetector::preProcess(const cv::Mat& image) {
    // 修改为CV_64F精度
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_64F);

    // 计算 DCT
    cv::Mat dctCoeffs = calculateDCT(floatImage);

    // --- Z字形遍历并置零 AC 系数 ---
    int rows = dctCoeffs.rows;
    int cols = dctCoeffs.cols;
    std::vector<double*> acCoeffPtrs;
    std::vector<double> acCoeffValues;

    int r = 0, c = 0;
    bool up = true;
    for (int i = 1; i < rows * cols; ++i) {
        if (r != 0 || c != 0) {
             if (dctCoeffs.at<double>(r, c) != 0.0) {
                 acCoeffPtrs.push_back(&dctCoeffs.at<double>(r, c));
                 acCoeffValues.push_back(dctCoeffs.at<double>(r, c));
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

    int numNonZeroAC = acCoeffPtrs.size();
    int numToZero = static_cast<int>(numNonZeroAC * 0.9);
    std::reverse(acCoeffPtrs.begin(), acCoeffPtrs.end());
    for (int i = 0; i < numToZero && i < acCoeffPtrs.size(); ++i) {
        *(acCoeffPtrs[i]) = 0.0;
    }
    // --- Z字形处理结束 ---

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
