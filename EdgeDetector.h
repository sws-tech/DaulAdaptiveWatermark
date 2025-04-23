#ifndef EDGE_DETECTOR_H
#define EDGE_DETECTOR_H

#include <opencv2/opencv.hpp>

class EdgeDetector {
public:
    // 构造函数，可以传入参数如 Canny 阈值，后处理阈值 t
    EdgeDetector(double lowThresh = 50, double highThresh = 150, double postProcessThresh = 20.0);

    // 执行边缘检测 (对应 Step 1)
    // 返回精确边缘图像 (二值图, 边缘为255, 非边缘为0)
    cv::Mat detectEdges(const cv::Mat& originalImage);

private:
    // 预处理：DCT域噪声抑制
    cv::Mat preProcess(const cv::Mat& image);

    // 后处理：去除误检边缘
    cv::Mat postProcess(const cv::Mat& edgeImage, const cv::Mat& originalImage);

    double cannyLowThreshold;
    double cannyHighThreshold;
    double postProcessingThreshold; // 阈值 t
};

#endif // EDGE_DETECTOR_H
