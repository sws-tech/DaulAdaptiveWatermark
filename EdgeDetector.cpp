#include "EdgeDetector.h"
#include "utils.h" // ��Ҫ DCT/IDCT

EdgeDetector::EdgeDetector(double lowThresh, double highThresh, double postProcessThresh)
    : cannyLowThreshold(lowThresh), cannyHighThreshold(highThresh), postProcessingThreshold(postProcessThresh) {}

cv::Mat EdgeDetector::detectEdges(const cv::Mat& originalImage) {
    if (originalImage.empty() || originalImage.channels() != 1) {
        throw std::runtime_error("EdgeDetector: Input image must be a single-channel grayscale image.");
    }

    // Step 1.1: Ԥ����
    cv::Mat preprocessedImage = preProcess(originalImage);

    // Step 1.2: Canny ��Ե���
    cv::Mat cannyEdges;
    cv::Canny(preprocessedImage, cannyEdges, cannyLowThreshold, cannyHighThreshold);

    cv::imwrite("canny_process.png", cannyEdges);

    // Step 1.3: ����
    cv::Mat finalEdges = postProcess(cannyEdges, originalImage); // ע�⣺����ʹ��ԭʼͼ�����ҶȲ�

    cv::imwrite("post_process.png", finalEdges);
    return finalEdges;
}

// Ԥ����DCT����������
cv::Mat EdgeDetector::preProcess(const cv::Mat& image) {
    // �޸�ΪCV_64F����
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_64F);

    // ���� DCT
    cv::Mat dctCoeffs = calculateDCT(floatImage);

    // --- Z���α��������� AC ϵ�� ---
    int rows = dctCoeffs.rows;
    int cols = dctCoeffs.cols;
    std::vector<double*> acCoeffPtrs;
    std::vector<double> acCoeffValues;

    int r = 0, c = 0;
    bool up = true;
    for (int i = 1; i < rows * cols; ++i) {
        if (r != 0 || c != 0) {
             if (dctCoeffs.at<double>(r, c) != 0.0) {
                 acCoeffPtrs.push_back(&dctCoeffs.at<double>(r, c));
                 acCoeffValues.push_back(dctCoeffs.at<double>(r, c));
             }
        }
        if (up) {
            if (r > 0 && c < cols - 1) { r--; c++; }
            else {
                up = false;
                if (c < cols - 1) c++; else r++;
            }
        } else {
            if (c > 0 && r < rows - 1) { c--; r++; }
            else {
                up = true;
                if (r < rows - 1) r++; else c++;
            }
        }
        if (r >= rows || c >= cols) break;
    }

    int numNonZeroAC = acCoeffPtrs.size();
    int numToZero = static_cast<int>(numNonZeroAC * 0.9);
    std::reverse(acCoeffPtrs.begin(), acCoeffPtrs.end());
    for (int i = 0; i < numToZero && i < acCoeffPtrs.size(); ++i) {
        *(acCoeffPtrs[i]) = 0.0;
    }
    // --- Z���δ������ ---

    // ���� IDCT
    cv::Mat idctResult = calculateIDCT(dctCoeffs);

    // ת���� 8λ�޷����������ͣ������нض�
    cv::Mat processedImage;
    idctResult.convertTo(processedImage, CV_8U);
    cv::normalize(processedImage, processedImage, 0, 255, cv::NORM_MINMAX);

    return processedImage;
}

// ����ȥ������Ե (ʽ 1)
cv::Mat EdgeDetector::postProcess(const cv::Mat& edgeImage, const cv::Mat& originalImage) {
    cv::Mat processedEdges = edgeImage.clone(); // �������������޸�

    for (int r = 1; r < edgeImage.rows - 1; ++r) {
        for (int c = 1; c < edgeImage.cols - 1; ++c) {
            // ֻ���� Canny ��⵽�ı�Ե���� (ֵΪ 255)
            if (edgeImage.at<uchar>(r, c) == 255) {
                double sumDiff = 0.0;
                uchar centerPixelValue = originalImage.at<uchar>(r, c);

                // �����������ĻҶȲ����ֵ֮��
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (dr == 0 && dc == 0) continue; // ������������
                        sumDiff += std::abs(static_cast<int>(originalImage.at<uchar>(r + dr, c + dc)) - static_cast<int>(centerPixelValue));
                    }
                }

                // ����ƽ���ҶȲ� A_diff
                double avgDiff = sumDiff / 8.0;

                // ���ƽ����С����ֵ t������Ϊ����죬��Ϊ 0
                if (avgDiff < postProcessingThreshold) {
                    processedEdges.at<uchar>(r, c) = 0;
                }
            }
        }
    }
    return processedEdges;
}
