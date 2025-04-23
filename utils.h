#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp> // ����ʹ��OpenCV

// ��������ṹ��
struct Region {
    cv::Rect bounds; // ����߽� (x, y, width, height)
    cv::Point center; // �������ĵ� (u, v)
    double score = 0.0; // �ۺϵ÷�
    // ������������÷� E, H, G, P
    double edgeScore = 0.0;
    double textureScore = 0.0;
    double grayScore = 0.0;
    double positionScore = 0.0;

    // ���ڷ��ص������ж�
    bool overlaps(const Region& other) const {
        return (bounds & other.bounds).area() > 0;
    }
};

// ����ͼ�����Ϣ
struct ImageBlock {
    cv::Rect bounds;
    bool isEdgeBlock = false;
    int edgePixelCount = 0; // N_xy
    int fixedEdgePixelCount = 0; // N*_xy
    double embeddingStrength = 0.0; // sigma_xy
    double totalModification = 0.0; // g(sigma_xy, w_xy)
    cv::Mat modificationWeights; // theta_xy(i,j) for edge blocks
};


// --- ������������ ---

// ����DCT����ɢ���ұ任�� - ��ʾ��
cv::Mat calculateDCT(const cv::Mat& input);

// ����IDCT������ɢ���ұ任�� - ��ʾ��
cv::Mat calculateIDCT(const cv::Mat& input);

// ������Ϣ��
double calculateEntropy(const cv::Mat& region);

// �����˹Ȩ��
cv::Mat calculateGaussianWeights(int rows, int cols, double sigma);


#endif // UTILS_H
