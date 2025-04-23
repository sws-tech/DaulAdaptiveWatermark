#ifndef WATERMARK_EMBEDDER_H
#define WATERMARK_EMBEDDER_H

#include "EdgeDetector.h"
#include "RegionSelector.h"
#include "WatermarkEncoder.h"
#include "BlockProcessor.h"
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class WatermarkEmbedder {
public:
    // ���캯������ʼ���������
    WatermarkEmbedder(int numRegions = 10, int edgeThreshold = 3); // ʾ������

    // ִ��������ˮӡǶ�����
    cv::Mat embedWatermark(const cv::Mat& originalImage, const std::string& watermarkText);

private:
    EdgeDetector edgeDetector;
    RegionScorer regionScorer; // RegionSelector �ڲ����õ�
    RegionSelector regionSelector;
    WatermarkEncoder watermarkEncoder;
    BlockProcessor blockProcessor;

    int numberOfRegions; // d
};

#endif // WATERMARK_EMBEDDER_H
