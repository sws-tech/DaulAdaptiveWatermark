#include "BlockProcessor.h"
#include <cmath> // for std::round, std::sqrt, std::pow, std::exp
#include <stdexcept>
#include <numeric> // for std::accumulate
#include "utils.h" // 需要 calculateGaussianWeights

BlockProcessor::BlockProcessor(int edgeThreshold, double gaussianSigma)
    : edgeBlockThreshold(edgeThreshold), gaussSigma(gaussianSigma) {
    if (edgeThreshold < 0) {
        throw std::invalid_argument("Edge block threshold cannot be negative.");
    }
     if (gaussianSigma <= 0) {
         throw std::invalid_argument("Gaussian sigma must be positive.");
     }
}

// 计算块的边缘像素数量 N_xy (式 8)
int BlockProcessor::countEdgePixels(const cv::Mat& blockEdgePatch) {
    // 边缘图像是二值的 (0 或 255)，直接统计非零像素
    return cv::countNonZero(blockEdgePatch);
}

// 确定固定边缘像素数量 N*_xy (式 12)
int BlockProcessor::determineFixedEdgeCount(int actualEdgeCount) {
    return (actualEdgeCount <= edgeBlockThreshold) ? 0 : edgeBlockThreshold;
    // 原文是 N*_xy = Th (边缘块), N*_xy = 0 (非边缘块)
    // 这里使用 Th 作为边缘块的固定值
}

// 计算嵌入强度 sigma_xy (式 11)
double BlockProcessor::calculateEmbeddingStrength(int fixedEdgeCount) {
    // σ = 0.2243 * N* + 1.5228
    return 0.2243 * static_cast<double>(fixedEdgeCount) + 1.5228;
}

// 划分区域为块，并计算块参数 (对应 Step 4)
std::vector<ImageBlock> BlockProcessor::prepareBlocks(const cv::Mat& regionPatch, const cv::Mat& regionEdgePatch, int watermarkLength) {
    if (regionPatch.empty() || regionEdgePatch.empty() || regionPatch.size() != regionEdgePatch.size()) {
        throw std::runtime_error("BlockProcessor: Input patches for block preparation are invalid or mismatched.");
    }
     if (watermarkLength <= 0) {
         throw std::invalid_argument("BlockProcessor: Watermark length must be positive.");
     }

    int regionRows = regionPatch.rows;
    int regionCols = regionPatch.cols;

    // --- 确定块的大小 ---
    // 尝试找到最接近正方形的块划分方式来容纳 watermarkLength 个块
    int bestRows = 1, bestCols = watermarkLength;
    double minAspectRatioDiff = std::abs(static_cast<double>(regionRows) / bestRows - static_cast<double>(regionCols) / bestCols);

    for (int r = 2; r * r <= watermarkLength; ++r) {
        if (watermarkLength % r == 0) {
            int c = watermarkLength / r;
            // 尝试 r 行 c 列
            double currentDiff = std::abs(static_cast<double>(regionRows) / r - static_cast<double>(regionCols) / c);
            if (currentDiff < minAspectRatioDiff) {
                minAspectRatioDiff = currentDiff;
                bestRows = r;
                bestCols = c;
            }
            // 尝试 c 行 r 列
             currentDiff = std::abs(static_cast<double>(regionRows) / c - static_cast<double>(regionCols) / r);
             if (currentDiff < minAspectRatioDiff) {
                 minAspectRatioDiff = currentDiff;
                 bestRows = c;
                 bestCols = r;
             }
        }
    }
    int numBlockRows = bestRows;
    int numBlockCols = bestCols;

    // 计算每个块的精确大小 (可能需要处理不能整除的情况，这里简化为整数除法)
    int blockHeight = regionRows / numBlockRows;
    int blockWidth = regionCols / numBlockCols;

    if (blockHeight == 0 || blockWidth == 0) {
         throw std::runtime_error("BlockProcessor: Region size is too small to be divided into blocks.");
    }

    std::vector<ImageBlock> blocks;
    blocks.reserve(watermarkLength);

    int blockIndex = 0;
    for (int br = 0; br < numBlockRows && blockIndex < watermarkLength; ++br) {
        for (int bc = 0; bc < numBlockCols && blockIndex < watermarkLength; ++bc) {
            ImageBlock block;
            // 计算块的边界 (注意边界处理)
            int startY = br * blockHeight;
            int startX = bc * blockWidth;
            int currentBlockHeight = (br == numBlockRows - 1) ? (regionRows - startY) : blockHeight; // 最后一行/列可能不同
            int currentBlockWidth = (bc == numBlockCols - 1) ? (regionCols - startX) : blockWidth;

            block.bounds = cv::Rect(startX, startY, currentBlockWidth, currentBlockHeight);

            // 提取块对应的边缘图部分
            cv::Mat blockEdgePatch = regionEdgePatch(block.bounds);

            // 计算 N_xy
            block.edgePixelCount = countEdgePixels(blockEdgePatch);

            // 计算 N*_xy 并判断是否为边缘块
            block.fixedEdgePixelCount = determineFixedEdgeCount(block.edgePixelCount);
            block.isEdgeBlock = (block.fixedEdgePixelCount > 0); // 或者 (block.edgePixelCount > edgeBlockThreshold)

            // 计算 sigma_xy
            block.embeddingStrength = calculateEmbeddingStrength(block.fixedEdgePixelCount);

            // 如果是边缘块，预计算高斯权重 (式 17)
            if (block.isEdgeBlock) {
                block.modificationWeights = calculateGaussianWeights(block.bounds.height, block.bounds.width, gaussSigma);
            }

            blocks.push_back(block);
            blockIndex++;
        }
    }
     if (blocks.size() != watermarkLength) {
         // 这通常不应该发生，除非块划分逻辑有误
         throw std::runtime_error("BlockProcessor: Number of prepared blocks does not match watermark length.");
     }

    return blocks;
}


// 计算 DC 系数 R_DCxy (式 13)
double BlockProcessor::calculateDCCoefficient(const cv::Mat& blockPatch) {
    if (blockPatch.empty()) return 0.0;
    // 确保是单通道
    if (blockPatch.channels() != 1) {
         throw std::runtime_error("Cannot calculate DC coefficient for multi-channel image.");
    }
    // 计算像素平均值
    cv::Scalar meanValue = cv::mean(blockPatch);
    double avgPixelValue = meanValue[0];

    // R_DC = (1 / sqrt(ab)) * sum(f_xy(i,j)) = (1 / sqrt(ab)) * (ab * avg) = sqrt(ab) * avg
    int a = blockPatch.rows;
    int b = blockPatch.cols;
    return std::sqrt(static_cast<double>(a * b)) * avgPixelValue;
}

// 计算量化后的 DC 系数 R*_DCxy (式 14)
double BlockProcessor::calculateQuantizedDCCoefficient(double dcCoefficient, double sigma_xy, int blockWidth, int blockHeight, int watermarkBit) {
    if (sigma_xy <= 0 || blockWidth <= 0 || blockHeight <= 0) {
        // 避免除零或无效参数
        return dcCoefficient; // 或者抛出异常
    }
    if (watermarkBit != 0 && watermarkBit != 1) {
         throw std::invalid_argument("Watermark bit must be 0 or 1.");
    }

    double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
    double quantizationStep = sigma_xy * ab_sqrt;

    if (quantizationStep < 1e-9) { // 避免除以非常小的值
        return dcCoefficient;
    }

    double normalizedDC = dcCoefficient / quantizationStep;
    double roundedDC = std::round(normalizedDC); // 四舍五入
    

    // (round(DC/step) + w) mod 2
    // 注意：原文公式似乎有误，(sigma(ab)^0.5 + w) mod 2 应该与量化决策有关
    // 常见的 QIM 形式是根据 w 选择量化区间的中点
    // 这里采用一种常见的 QIM 实现：
    // 如果 w=0，量化到最近的偶数倍半步长；如果 w=1，量化到最近的奇数倍半步长。
    double quantizedDC;
    //if (watermarkBit == 0) { // 嵌入 0
    //    // 量化到 k * step
    //    quantizedDC = roundedDC * quantizationStep;
    //} else { // 嵌入 1
    //    // 量化到 (k + 0.5) * step 或 (k - 0.5) * step，取决于哪个更近
    //    if (normalizedDC > roundedDC) { // 在两个整数中间的右侧
    //         quantizedDC = (roundedDC + 0.5) * quantizationStep;
    //    } else { // 在两个整数中间的左侧或正好在整数上
    //         quantizedDC = (roundedDC - 0.5) * quantizationStep;
    //    }
         // 确保量化后的值与原始值符号相同（可选，但通常需要）
         // if ((quantizedDC * dcCoefficient) < 0 && std::abs(dcCoefficient) > 1e-9) {
         //     quantizedDC = (roundedDC + (normalizedDC > roundedDC ? 0.5 : -0.5)) * quantizationStep;
         //     // 再次检查，如果还是符号相反，可能需要特殊处理或调整逻辑
         // }
    //}


    // --- 另一种解释原文公式 (14) 的方式 ---
    int term = static_cast<int>(std::round(normalizedDC) + watermarkBit);
    if (term % 2 == 1) { // 对应原文第一种情况
        quantizedDC = (std::round(normalizedDC) - 0.5) * quantizationStep;
    }
    else { // 对应原文第二种情况
        quantizedDC = (std::round(normalizedDC) + 0.5) * quantizationStep;
    }
    // --- 解释结束 ---
    // 我们将使用上面更常见的 QIM 实现。

    return quantizedDC;
}


// 计算总修改量 g(sigma_xy, w_xy) (式 15)
double BlockProcessor::calculateTotalModification(double dcCoefficient, double quantizedDCCoefficient, int blockWidth, int blockHeight) {
    double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
    return ab_sqrt * (quantizedDCCoefficient - dcCoefficient);
}

// 分配像素修改量 w_xy(i,j) (式 16)
cv::Mat BlockProcessor::distributeModification(const ImageBlock& block, double totalModification, const cv::Mat& gaussianWeights) {
    int rows = block.bounds.height;
    int cols = block.bounds.width;
    cv::Mat modificationMatrix = cv::Mat::zeros(rows, cols, CV_64F); // 使用 double 存储修改量

    if (rows == 0 || cols == 0) return modificationMatrix;

    if (!block.isEdgeBlock) {
        // 非边缘块：平均分配
        double modificationPerPixel = totalModification / (rows * cols);
        modificationMatrix.setTo(cv::Scalar(modificationPerPixel));
    } else {
        // 边缘块：按高斯权重分配
        if (gaussianWeights.empty() || gaussianWeights.size() != cv::Size(cols, rows) || gaussianWeights.type() != CV_64F) {
             throw std::runtime_error("Invalid Gaussian weights provided for edge block modification distribution.");
        }
        // w_xy(i, j) = theta_xy(i, j) * g
        modificationMatrix = gaussianWeights * totalModification;
    }

    return modificationMatrix;
}


// 计算单个块的像素修改量 (对应 Step 5)
cv::Mat BlockProcessor::calculatePixelModifications(const ImageBlock& block, const cv::Mat& blockPatch, int watermarkBit) {
     if (blockPatch.empty() || blockPatch.size() != block.bounds.size()) {
         throw std::runtime_error("Block patch size mismatch in calculatePixelModifications.");
     }
     if (blockPatch.channels() != 1) {
         throw std::runtime_error("Block patch must be single-channel grayscale.");
     }

    // 1. 计算原始 DC 系数 (式 13)
    cv::Mat floatPatch;
    blockPatch.convertTo(floatPatch, CV_32F); // 需要浮点数进行计算
    double dcCoefficient = calculateDCCoefficient(floatPatch);

    // 2. 计算量化后的 DC 系数 (式 14)
    double quantizedDCCoefficient = calculateQuantizedDCCoefficient(dcCoefficient, block.embeddingStrength, block.bounds.width, block.bounds.height, watermarkBit);

    // 3. 计算总修改量 g (式 15)
    double totalModification = calculateTotalModification(dcCoefficient, quantizedDCCoefficient, block.bounds.width, block.bounds.height);

    // 4. 分配像素修改量 w_xy(i,j) (式 16)
    cv::Mat modificationMatrix = distributeModification(block, totalModification, block.modificationWeights); // modificationWeights 在 prepareBlocks 中已计算

    return modificationMatrix; // 返回的是 double 类型的修改量矩阵
}

// 新增：实现 processRegionAsBlock
ImageBlock BlockProcessor::processRegionAsBlock(const cv::Mat& blockPatch, const cv::Mat& blockEdgePatch, const cv::Rect& blockBounds) {
    if (blockPatch.empty() || blockEdgePatch.empty() || blockPatch.size() != blockEdgePatch.size()) {
        throw std::runtime_error("BlockProcessor: Input patches for single block processing are invalid or mismatched.");
    }
     if (blockPatch.size() != blockBounds.size()) {
         throw std::runtime_error("BlockProcessor: Patch size does not match block bounds.");
     }

    ImageBlock block;
    block.bounds = blockBounds;

    // Step 4: 计算块参数 (N_xy, N*_xy, isEdgeBlock, sigma_xy)
    block.edgePixelCount = countEdgePixels(blockEdgePatch); // 调用 private 方法
    block.fixedEdgePixelCount = determineFixedEdgeCount(block.edgePixelCount); // 调用 private 方法
    block.isEdgeBlock = (block.fixedEdgePixelCount > 0); // 或 (block.edgePixelCount > edgeBlockThreshold)
    block.embeddingStrength = calculateEmbeddingStrength(block.fixedEdgePixelCount); // 调用 private 方法

    // 如果是边缘块，计算高斯权重
    if (block.isEdgeBlock) {
        // 调用 utils.h 中的全局函数，并使用内部的 gaussSigma
        block.modificationWeights = calculateGaussianWeights(block.bounds.height, block.bounds.width, this->gaussSigma); // 内部访问 private gaussSigma
    }

    // totalModification 会在 calculatePixelModifications 中计算，这里初始化
    block.totalModification = 0.0;

    return block;
}
