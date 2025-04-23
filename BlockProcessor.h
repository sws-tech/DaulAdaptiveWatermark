#ifndef BLOCK_PROCESSOR_H
#define BLOCK_PROCESSOR_H

#include "utils.h"
#include <vector>
#include <opencv2/opencv.hpp>

class BlockProcessor {
public:
    // 构造函数，传入划分阈值 Th 和高斯滤波标准差
    BlockProcessor(int edgeThreshold = 3, double gaussianSigma = 1.5); // Th=5 示例

    // 划分区域为块，并计算块参数 (对应 Step 4)
    std::vector<ImageBlock> prepareBlocks(const cv::Mat& regionPatch, const cv::Mat& regionEdgePatch, int watermarkLength);

    // 新增：处理单个区域/块，计算其属性 (用于简化实现)
    ImageBlock processRegionAsBlock(const cv::Mat& blockPatch, const cv::Mat& blockEdgePatch, const cv::Rect& blockBounds);

    // 计算单个块的像素修改量 (对应 Step 5)
    // 返回每个像素的修改量矩阵 w_xy(i,j)
    cv::Mat calculatePixelModifications(const ImageBlock& block, const cv::Mat& blockPatch, int watermarkBit);

    // 确保 DC 计算是 public 的
    double calculateDCCoefficient(const cv::Mat& blockPatch);

private:
    int edgeBlockThreshold; // Th
    double gaussSigma; // 高斯窗标准差

    // 计算块的边缘像素数量 N_xy (式 8)
    int countEdgePixels(const cv::Mat& blockEdgePatch);

    // 确定固定边缘像素数量 N*_xy (式 12)
    int determineFixedEdgeCount(int actualEdgeCount);

    // 计算嵌入强度 sigma_xy (式 11)
    double calculateEmbeddingStrength(int fixedEdgeCount);

    // 计算 DC 系数 R_DCxy (式 13)
    // double calculateDCCoefficient(const cv::Mat& blockPatch); // 如果之前是 private，移到 public

    // 计算量化后的 DC 系数 R*_DCxy (式 14)
    double calculateQuantizedDCCoefficient(double dcCoefficient, double sigma_xy, int blockWidth, int blockHeight, int watermarkBit);

    // 计算总修改量 g(sigma_xy, w_xy) (式 15)
    double calculateTotalModification(double dcCoefficient, double quantizedDCCoefficient, int blockWidth, int blockHeight);

    // 分配像素修改量 w_xy(i,j) (式 16)
    cv::Mat distributeModification(const ImageBlock& block, double totalModification, const cv::Mat& gaussianWeights);
};

#endif // BLOCK_PROCESSOR_H
