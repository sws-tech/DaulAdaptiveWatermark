#ifndef WATERMARK_EXTRACTOR_H
#define WATERMARK_EXTRACTOR_H

#include "EdgeDetector.h"
#include "RegionSelector.h"
#include "BlockProcessor.h"
#include "WatermarkDecoder.h" // ��������������
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class WatermarkExtractor {
public:
    // ���캯��������ԭʼͼ��·������
    WatermarkExtractor(int expectedWatermarkLength, const std::string& rawImagePath, int edgeThreshold = 10);

    // ִ��������ˮӡ��ȡ����
    std::string extractWatermark(const cv::Mat& watermarkedImage);

private:
    EdgeDetector edgeDetector;
    RegionScorer regionScorer; // RegionSelector �ڲ����õ�
    RegionSelector regionSelector;
    BlockProcessor blockProcessor;
    WatermarkDecoder watermarkDecoder; // ����������ʵ��

    int expectedWatermarkLength; // m
    std::string rawImagePath; // ������ԭʼͼ��·��
};

#endif // WATERMARK_EXTRACTOR_H
