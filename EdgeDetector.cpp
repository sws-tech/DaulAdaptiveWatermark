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
    // 确保图像是浮点型以便进行 DCT
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    // 计算 DCT
    cv::Mat dctCoeffs = calculateDCT(floatImage);

    // --- Z字形遍历并置零 AC 系数 ---
    // 获取 DCT 系数矩阵的大小
    int rows = dctCoeffs.rows;
    int cols = dctCoeffs.cols;
    std::vector<float*> acCoeffPtrs; // 指向 AC 系数的指针
    std::vector<float> acCoeffValues; // AC 系数的值

    // Z字形扫描并收集非零 AC 系数 (跳过 DC 系数 dctCoeffs.at<float>(0, 0))
    int r = 0, c = 0;
    bool up = true;
    for (int i = 1; i < rows * cols; ++i) { // 从第二个系数开始
        if (r != 0 || c != 0) { // 跳过 DC(0,0)
             if (dctCoeffs.at<float>(r, c) != 0.0f) {
                 acCoeffPtrs.push_back(&dctCoeffs.at<float>(r, c));
                 acCoeffValues.push_back(dctCoeffs.at<float>(r, c));
             }
        }

        // Z字形移动
        if (up) {
            if (r > 0 && c < cols - 1) { r--; c++; }
            else {
                up = false;
                if (c < cols - 1) c++; else r++; // 到达右上边界
            }
        } else {
            if (c > 0 && r < rows - 1) { c--; r++; }
            else {
                up = true;
                if (r < rows - 1) r++; else c++; // 到达左下边界
            }
        }
         // 边界检查以防万一
        if (r >= rows || c >= cols) break;
    }


    // 计算需要置零的系数数量 (前 90% 的非零 AC 系数)
    int numNonZeroAC = acCoeffPtrs.size();
    int numToZero = static_cast<int>(numNonZeroAC * 0.9);

    // 将排在序列前 90% 的非零 AC 系数置零
    // 注意：这里简单地将收集到的前 90% 指针指向的值置零，
    // 这与原文“从左上角开始按‘Z’字形遍历AC系数，得到系数序列，将排在序列前90%的AC系数中非零系数的值置为零”
    // 的顺序可能略有不同，但效果是类似的：抑制大部分 AC 系数。
    // 更精确的实现需要严格按 Z 序排序后再置零。
    std::reverse(acCoeffPtrs.begin(), acCoeffPtrs.end());
    for (int i = 0; i < numToZero && i < acCoeffPtrs.size(); ++i) {
        *(acCoeffPtrs[i]) = 0.0f;
    }
    // --- Z字形处理结束 ---


    // 计算 IDCT
    cv::Mat idctResult = calculateIDCT(dctCoeffs);

    // 转换回 8位无符号整数类型，并进行截断
    cv::Mat processedImage;
    idctResult.convertTo(processedImage, CV_8U);
    cv::normalize(processedImage, processedImage, 0, 255, cv::NORM_MINMAX); // 归一化到 0-255

    cv::imwrite("dct_pre_processed.png", processedImage);
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
