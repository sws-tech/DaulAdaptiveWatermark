#ifndef EDGE_DETECTOR_H
#define EDGE_DETECTOR_H

#include <opencv2/opencv.hpp>

class EdgeDetector {
public:
    // ���캯�������Դ�������� Canny ��ֵ��������ֵ t
    EdgeDetector(double lowThresh = 50, double highThresh = 150, double postProcessThresh = 20.0);

    // ִ�б�Ե��� (��Ӧ Step 1)
    // ���ؾ�ȷ��Եͼ�� (��ֵͼ, ��ԵΪ255, �Ǳ�ԵΪ0)
    cv::Mat detectEdges(const cv::Mat& originalImage);

private:
    // Ԥ����DCT����������
    cv::Mat preProcess(const cv::Mat& image);

    // ����ȥ������Ե
    cv::Mat postProcess(const cv::Mat& edgeImage, const cv::Mat& originalImage);

    double cannyLowThreshold;
    double cannyHighThreshold;
    double postProcessingThreshold; // ��ֵ t
};

#endif // EDGE_DETECTOR_H
