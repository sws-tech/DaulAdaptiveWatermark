#include "BlockProcessor.h"
#include <cmath> // for std::round, std::sqrt, std::pow, std::exp
#include <stdexcept>
#include <numeric> // for std::accumulate
#include "utils.h" // ��Ҫ calculateGaussianWeights

BlockProcessor::BlockProcessor(int edgeThreshold, double gaussianSigma)
    : edgeBlockThreshold(edgeThreshold), gaussSigma(gaussianSigma) {
    if (edgeThreshold < 0) {
        throw std::invalid_argument("Edge block threshold cannot be negative.");
    }
     if (gaussianSigma <= 0) {
         throw std::invalid_argument("Gaussian sigma must be positive.");
     }
}

// �����ı�Ե�������� N_xy (ʽ 8)
int BlockProcessor::countEdgePixels(const cv::Mat& blockEdgePatch) {
    // ��Եͼ���Ƕ�ֵ�� (0 �� 255)��ֱ��ͳ�Ʒ�������
    return cv::countNonZero(blockEdgePatch);
}

// ȷ���̶���Ե�������� N*_xy (ʽ 12)
int BlockProcessor::determineFixedEdgeCount(int actualEdgeCount) {
    return (actualEdgeCount <= edgeBlockThreshold) ? 0 : edgeBlockThreshold;
    // ԭ���� N*_xy = Th (��Ե��), N*_xy = 0 (�Ǳ�Ե��)
    // ����ʹ�� Th ��Ϊ��Ե��Ĺ̶�ֵ
}

// ����Ƕ��ǿ�� sigma_xy (ʽ 11)
double BlockProcessor::calculateEmbeddingStrength(int fixedEdgeCount) {
    // �� = 0.2243 * N* + 1.5228
    return 0.2243 * static_cast<double>(fixedEdgeCount) + 1.5228;
}

// ��������Ϊ�飬���������� (��Ӧ Step 4)
std::vector<ImageBlock> BlockProcessor::prepareBlocks(const cv::Mat& regionPatch, const cv::Mat& regionEdgePatch, int watermarkLength) {
    if (regionPatch.empty() || regionEdgePatch.empty() || regionPatch.size() != regionEdgePatch.size()) {
        throw std::runtime_error("BlockProcessor: Input patches for block preparation are invalid or mismatched.");
    }
     if (watermarkLength <= 0) {
         throw std::invalid_argument("BlockProcessor: Watermark length must be positive.");
     }

    int regionRows = regionPatch.rows;
    int regionCols = regionPatch.cols;

    // --- ȷ����Ĵ�С ---
    // �����ҵ���ӽ������εĿ黮�ַ�ʽ������ watermarkLength ����
    int bestRows = 1, bestCols = watermarkLength;
    double minAspectRatioDiff = std::abs(static_cast<double>(regionRows) / bestRows - static_cast<double>(regionCols) / bestCols);

    for (int r = 2; r * r <= watermarkLength; ++r) {
        if (watermarkLength % r == 0) {
            int c = watermarkLength / r;
            // ���� r �� c ��
            double currentDiff = std::abs(static_cast<double>(regionRows) / r - static_cast<double>(regionCols) / c);
            if (currentDiff < minAspectRatioDiff) {
                minAspectRatioDiff = currentDiff;
                bestRows = r;
                bestCols = c;
            }
            // ���� c �� r ��
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

    // ����ÿ����ľ�ȷ��С (������Ҫ����������������������Ϊ��������)
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
            // �����ı߽� (ע��߽紦��)
            int startY = br * blockHeight;
            int startX = bc * blockWidth;
            int currentBlockHeight = (br == numBlockRows - 1) ? (regionRows - startY) : blockHeight; // ���һ��/�п��ܲ�ͬ
            int currentBlockWidth = (bc == numBlockCols - 1) ? (regionCols - startX) : blockWidth;

            block.bounds = cv::Rect(startX, startY, currentBlockWidth, currentBlockHeight);

            // ��ȡ���Ӧ�ı�Եͼ����
            cv::Mat blockEdgePatch = regionEdgePatch(block.bounds);

            // ���� N_xy
            block.edgePixelCount = countEdgePixels(blockEdgePatch);

            // ���� N*_xy ���ж��Ƿ�Ϊ��Ե��
            block.fixedEdgePixelCount = determineFixedEdgeCount(block.edgePixelCount);
            block.isEdgeBlock = (block.fixedEdgePixelCount > 0); // ���� (block.edgePixelCount > edgeBlockThreshold)

            // ���� sigma_xy
            block.embeddingStrength = calculateEmbeddingStrength(block.fixedEdgePixelCount);

            // ����Ǳ�Ե�飬Ԥ�����˹Ȩ�� (ʽ 17)
            if (block.isEdgeBlock) {
                block.modificationWeights = calculateGaussianWeights(block.bounds.height, block.bounds.width, gaussSigma);
            }

            blocks.push_back(block);
            blockIndex++;
        }
    }
     if (blocks.size() != watermarkLength) {
         // ��ͨ����Ӧ�÷��������ǿ黮���߼�����
         throw std::runtime_error("BlockProcessor: Number of prepared blocks does not match watermark length.");
     }

    return blocks;
}


// ���� DC ϵ�� R_DCxy (ʽ 13)
double BlockProcessor::calculateDCCoefficient(const cv::Mat& blockPatch) {
    if (blockPatch.empty()) return 0.0;
    // ȷ���ǵ�ͨ��
    if (blockPatch.channels() != 1) {
         throw std::runtime_error("Cannot calculate DC coefficient for multi-channel image.");
    }
    // ��������ƽ��ֵ
    cv::Scalar meanValue = cv::mean(blockPatch);
    double avgPixelValue = meanValue[0];

    // R_DC = (1 / sqrt(ab)) * sum(f_xy(i,j)) = (1 / sqrt(ab)) * (ab * avg) = sqrt(ab) * avg
    int a = blockPatch.rows;
    int b = blockPatch.cols;
    return std::sqrt(static_cast<double>(a * b)) * avgPixelValue;
}

// ����������� DC ϵ�� R*_DCxy (ʽ 14)
double BlockProcessor::calculateQuantizedDCCoefficient(double dcCoefficient, double sigma_xy, int blockWidth, int blockHeight, int watermarkBit) {
    if (sigma_xy <= 0 || blockWidth <= 0 || blockHeight <= 0) {
        // ����������Ч����
        return dcCoefficient; // �����׳��쳣
    }
    if (watermarkBit != 0 && watermarkBit != 1) {
         throw std::invalid_argument("Watermark bit must be 0 or 1.");
    }

    double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
    double quantizationStep = sigma_xy * ab_sqrt;

    if (quantizationStep < 1e-9) { // ������Էǳ�С��ֵ
        return dcCoefficient;
    }

    double normalizedDC = dcCoefficient / quantizationStep;
    double roundedDC = std::round(normalizedDC); // ��������
    

    // (round(DC/step) + w) mod 2
    // ע�⣺ԭ�Ĺ�ʽ�ƺ�����(sigma(ab)^0.5 + w) mod 2 Ӧ�������������й�
    // ������ QIM ��ʽ�Ǹ��� w ѡ������������е�
    // �������һ�ֳ����� QIM ʵ�֣�
    // ��� w=0�������������ż�����벽������� w=1��������������������벽����
    double quantizedDC;
    //if (watermarkBit == 0) { // Ƕ�� 0
    //    // ������ k * step
    //    quantizedDC = roundedDC * quantizationStep;
    //} else { // Ƕ�� 1
    //    // ������ (k + 0.5) * step �� (k - 0.5) * step��ȡ�����ĸ�����
    //    if (normalizedDC > roundedDC) { // �����������м���Ҳ�
    //         quantizedDC = (roundedDC + 0.5) * quantizationStep;
    //    } else { // �����������м������������������
    //         quantizedDC = (roundedDC - 0.5) * quantizationStep;
    //    }
         // ȷ���������ֵ��ԭʼֵ������ͬ����ѡ����ͨ����Ҫ��
         // if ((quantizedDC * dcCoefficient) < 0 && std::abs(dcCoefficient) > 1e-9) {
         //     quantizedDC = (roundedDC + (normalizedDC > roundedDC ? 0.5 : -0.5)) * quantizationStep;
         //     // �ٴμ�飬������Ƿ����෴��������Ҫ���⴦�������߼�
         // }
    //}


    // --- ��һ�ֽ���ԭ�Ĺ�ʽ (14) �ķ�ʽ ---
    int term = static_cast<int>(std::round(normalizedDC) + watermarkBit);
    if (term % 2 == 1) { // ��Ӧԭ�ĵ�һ�����
        quantizedDC = (std::round(normalizedDC) - 0.5) * quantizationStep;
    }
    else { // ��Ӧԭ�ĵڶ������
        quantizedDC = (std::round(normalizedDC) + 0.5) * quantizationStep;
    }
    // --- ���ͽ��� ---
    // ���ǽ�ʹ������������� QIM ʵ�֡�

    return quantizedDC;
}


// �������޸��� g(sigma_xy, w_xy) (ʽ 15)
double BlockProcessor::calculateTotalModification(double dcCoefficient, double quantizedDCCoefficient, int blockWidth, int blockHeight) {
    double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
    return ab_sqrt * (quantizedDCCoefficient - dcCoefficient);
}

// ���������޸��� w_xy(i,j) (ʽ 16)
cv::Mat BlockProcessor::distributeModification(const ImageBlock& block, double totalModification, const cv::Mat& gaussianWeights) {
    int rows = block.bounds.height;
    int cols = block.bounds.width;
    cv::Mat modificationMatrix = cv::Mat::zeros(rows, cols, CV_64F); // ʹ�� double �洢�޸���

    if (rows == 0 || cols == 0) return modificationMatrix;

    if (!block.isEdgeBlock) {
        // �Ǳ�Ե�飺ƽ������
        double modificationPerPixel = totalModification / (rows * cols);
        modificationMatrix.setTo(cv::Scalar(modificationPerPixel));
    } else {
        // ��Ե�飺����˹Ȩ�ط���
        if (gaussianWeights.empty() || gaussianWeights.size() != cv::Size(cols, rows) || gaussianWeights.type() != CV_64F) {
             throw std::runtime_error("Invalid Gaussian weights provided for edge block modification distribution.");
        }
        // w_xy(i, j) = theta_xy(i, j) * g
        modificationMatrix = gaussianWeights * totalModification;
    }

    return modificationMatrix;
}


// ���㵥����������޸��� (��Ӧ Step 5)
cv::Mat BlockProcessor::calculatePixelModifications(const ImageBlock& block, const cv::Mat& blockPatch, int watermarkBit) {
     if (blockPatch.empty() || blockPatch.size() != block.bounds.size()) {
         throw std::runtime_error("Block patch size mismatch in calculatePixelModifications.");
     }
     if (blockPatch.channels() != 1) {
         throw std::runtime_error("Block patch must be single-channel grayscale.");
     }

    // 1. ����ԭʼ DC ϵ�� (ʽ 13)
    cv::Mat floatPatch;
    blockPatch.convertTo(floatPatch, CV_32F); // ��Ҫ���������м���
    double dcCoefficient = calculateDCCoefficient(floatPatch);

    // 2. ����������� DC ϵ�� (ʽ 14)
    double quantizedDCCoefficient = calculateQuantizedDCCoefficient(dcCoefficient, block.embeddingStrength, block.bounds.width, block.bounds.height, watermarkBit);

    // 3. �������޸��� g (ʽ 15)
    double totalModification = calculateTotalModification(dcCoefficient, quantizedDCCoefficient, block.bounds.width, block.bounds.height);

    // 4. ���������޸��� w_xy(i,j) (ʽ 16)
    cv::Mat modificationMatrix = distributeModification(block, totalModification, block.modificationWeights); // modificationWeights �� prepareBlocks ���Ѽ���

    return modificationMatrix; // ���ص��� double ���͵��޸�������
}

// ������ʵ�� processRegionAsBlock
ImageBlock BlockProcessor::processRegionAsBlock(const cv::Mat& blockPatch, const cv::Mat& blockEdgePatch, const cv::Rect& blockBounds) {
    if (blockPatch.empty() || blockEdgePatch.empty() || blockPatch.size() != blockEdgePatch.size()) {
        throw std::runtime_error("BlockProcessor: Input patches for single block processing are invalid or mismatched.");
    }
     if (blockPatch.size() != blockBounds.size()) {
         throw std::runtime_error("BlockProcessor: Patch size does not match block bounds.");
     }

    ImageBlock block;
    block.bounds = blockBounds;

    // Step 4: �������� (N_xy, N*_xy, isEdgeBlock, sigma_xy)
    block.edgePixelCount = countEdgePixels(blockEdgePatch); // ���� private ����
    block.fixedEdgePixelCount = determineFixedEdgeCount(block.edgePixelCount); // ���� private ����
    block.isEdgeBlock = (block.fixedEdgePixelCount > 0); // �� (block.edgePixelCount > edgeBlockThreshold)
    block.embeddingStrength = calculateEmbeddingStrength(block.fixedEdgePixelCount); // ���� private ����

    // ����Ǳ�Ե�飬�����˹Ȩ��
    if (block.isEdgeBlock) {
        // ���� utils.h �е�ȫ�ֺ�������ʹ���ڲ��� gaussSigma
        block.modificationWeights = calculateGaussianWeights(block.bounds.height, block.bounds.width, this->gaussSigma); // �ڲ����� private gaussSigma
    }

    // totalModification ���� calculatePixelModifications �м��㣬�����ʼ��
    block.totalModification = 0.0;

    return block;
}
