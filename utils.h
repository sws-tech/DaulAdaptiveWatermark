#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp> // 假设使用OpenCV

// 定义区域结构体
struct Region {
    cv::Rect bounds; // 区域边界 (x, y, width, height)
    cv::Point center; // 区域中心点 (u, v)
    double score = 0.0; // 综合得分
    // 可以添加其他得分 E, H, G, P
    double edgeScore = 0.0;
    double textureScore = 0.0;
    double grayScore = 0.0;
    double positionScore = 0.0;

    // 用于非重叠区域判断
    bool overlaps(const Region& other) const {
        return (bounds & other.bounds).area() > 0;
    }
};

// 定义图像块信息
struct ImageBlock {
    cv::Rect bounds;
    bool isEdgeBlock = false;
    int edgePixelCount = 0; // N_xy
    int fixedEdgePixelCount = 0; // N*_xy
    double embeddingStrength = 0.0; // sigma_xy
    double totalModification = 0.0; // g(sigma_xy, w_xy)
    cv::Mat modificationWeights; // theta_xy(i,j) for edge blocks
};


// --- 辅助函数声明 ---

// 计算DCT（离散余弦变换） - 简化示意
cv::Mat calculateDCT(const cv::Mat& input);

// 计算IDCT（逆离散余弦变换） - 简化示意
cv::Mat calculateIDCT(const cv::Mat& input);

// 计算信息熵
double calculateEntropy(const cv::Mat& region);

// 计算高斯权重
cv::Mat calculateGaussianWeights(int rows, int cols, double sigma);


#endif // UTILS_H
