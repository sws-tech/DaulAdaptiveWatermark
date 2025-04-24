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
    // ���캯����������Ҫԭʼͼ��·��
    WatermarkExtractor(int expectedWatermarkLength, int edgeThreshold = 25);

    // ִ��������ˮӡ��ȡ����
    std::string extractWatermark(const cv::Mat& watermarkedImage);

private:
    EdgeDetector edgeDetector;
    RegionScorer regionScorer; // RegionSelector �ڲ����õ�
    RegionSelector regionSelector;
    BlockProcessor blockProcessor;
    WatermarkDecoder watermarkDecoder; // ����������ʵ��

    int expectedWatermarkLength; // m
};

#endif // WATERMARK_EXTRACTOR_H
