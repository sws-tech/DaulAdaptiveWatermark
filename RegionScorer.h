#ifndef REGION_SCORER_H
#define REGION_SCORER_H

#include "utils.h"
#include <opencv2/opencv.hpp>

class RegionScorer {
public:
    // ���캯�������Դ���Ȩ�� alpha, beta, gamma, delta
    RegionScorer(double alpha = 0.4, double beta = 0.2, double gamma = 0.2, double delta = 0.2);

    // ���㵥��������ۺϵ÷� (��Ӧ Step 2 ���ּ���)
    void calculateRegionScores(Region& region, const cv::Mat& originalPatch, const cv::Mat& edgePatch, const cv::Point& imageCenter);

    // �����Ե�÷� E_uv (ʽ 2)
    double calculateEdgeScore(const cv::Mat& edgePatch);

    // ��������÷� H_uv (ʽ 3) - ʹ����Ϣ��
    double calculateTextureScore(const cv::Mat& originalPatch);

    // ����Ҷȵ÷� G_uv (ʽ 4)
    double calculateGrayScore(const cv::Mat& originalPatch);

    // ����λ�õ÷� P_uv (ʽ 5)
    double calculatePositionScore(const cv::Point& regionCenter, const cv::Point& imageCenter, int regionWidth, int regionHeight);

    // �����ۺϵ÷� Score_uv (ʽ 6)
    double calculateCombinedScore(double E, double H, double G, double P);

private:
    double weightAlpha; // alpha
    double weightBeta;  // beta
    double weightGamma; // gamma
    double weightDelta; // delta
};

#endif // REGION_SCORER_H
