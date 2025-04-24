#ifndef REGION_SELECTOR_H
#define REGION_SELECTOR_H

#include "utils.h"
#include "RegionScorer.h"
#include <vector>
#include <opencv2/opencv.hpp>

class RegionSelector {
public:
    // ���캯����������������Ŀ���������� d���������� a���������� b
    RegionSelector(RegionScorer scorer, int numRegionsToSelect = 10, double windowScale = 0.25, double stepScale = 0.25);

    // ѡ��Ƕ������ (��Ӧ Step 2 ��Ҫ�߼�)
    std::vector<Region> selectEmbeddingRegions(const cv::Mat& originalImage, const cv::Mat& edgeImage);

    // ���� Getter ����
    double getWindowScale() const { return windowSizeScale; }
    double getStepScale() const { return stepSizeScale; }

private:
    RegionScorer regionScorer;
    int targetRegionCount; // d
    double windowSizeScale; // a: ������С��ͼ��ߴ�ı���
    double stepSizeScale;   // b: �����봰�ڴ�С�ı���
};

#endif // REGION_SELECTOR_H
