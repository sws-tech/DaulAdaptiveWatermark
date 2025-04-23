#ifndef BLOCK_PROCESSOR_H
#define BLOCK_PROCESSOR_H

#include "utils.h"
#include <vector>
#include <opencv2/opencv.hpp>

class BlockProcessor {
public:
    // ���캯�������뻮����ֵ Th �͸�˹�˲���׼��
    BlockProcessor(int edgeThreshold = 3, double gaussianSigma = 1.5); // Th=5 ʾ��

    // ��������Ϊ�飬���������� (��Ӧ Step 4)
    std::vector<ImageBlock> prepareBlocks(const cv::Mat& regionPatch, const cv::Mat& regionEdgePatch, int watermarkLength);

    // ����������������/�飬���������� (���ڼ�ʵ��)
    ImageBlock processRegionAsBlock(const cv::Mat& blockPatch, const cv::Mat& blockEdgePatch, const cv::Rect& blockBounds);

    // ���㵥����������޸��� (��Ӧ Step 5)
    // ����ÿ�����ص��޸������� w_xy(i,j)
    cv::Mat calculatePixelModifications(const ImageBlock& block, const cv::Mat& blockPatch, int watermarkBit);

    // ȷ�� DC ������ public ��
    double calculateDCCoefficient(const cv::Mat& blockPatch);

private:
    int edgeBlockThreshold; // Th
    double gaussSigma; // ��˹����׼��

    // �����ı�Ե�������� N_xy (ʽ 8)
    int countEdgePixels(const cv::Mat& blockEdgePatch);

    // ȷ���̶���Ե�������� N*_xy (ʽ 12)
    int determineFixedEdgeCount(int actualEdgeCount);

    // ����Ƕ��ǿ�� sigma_xy (ʽ 11)
    double calculateEmbeddingStrength(int fixedEdgeCount);

    // ���� DC ϵ�� R_DCxy (ʽ 13)
    // double calculateDCCoefficient(const cv::Mat& blockPatch); // ���֮ǰ�� private���Ƶ� public

    // ����������� DC ϵ�� R*_DCxy (ʽ 14)
    double calculateQuantizedDCCoefficient(double dcCoefficient, double sigma_xy, int blockWidth, int blockHeight, int watermarkBit);

    // �������޸��� g(sigma_xy, w_xy) (ʽ 15)
    double calculateTotalModification(double dcCoefficient, double quantizedDCCoefficient, int blockWidth, int blockHeight);

    // ���������޸��� w_xy(i,j) (ʽ 16)
    cv::Mat distributeModification(const ImageBlock& block, double totalModification, const cv::Mat& gaussianWeights);
};

#endif // BLOCK_PROCESSOR_H
