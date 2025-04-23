#include "WatermarkEmbedder.h"
#include <stdexcept>
#include <iostream> // For warnings/errors
#include "utils.h" // ��Ҫ calculateGaussianWeights (���ڲ���Ҫֱ�ӵ�����)

WatermarkEmbedder::WatermarkEmbedder(int numRegions, int edgeThreshold)
    : edgeDetector(), // ʹ��Ĭ�ϲ��������ض�����
      regionScorer(), // ʹ��Ĭ��Ȩ�ػ����ض�Ȩ��
      regionSelector(regionScorer, numRegions), // ���� scorer ��Ŀ��������
      watermarkEncoder(), // ʹ��Ĭ�ϲ���
      blockProcessor(edgeThreshold), // �����Ե����ֵ Th
      numberOfRegions(numRegions)
{}

cv::Mat WatermarkEmbedder::embedWatermark(const cv::Mat& originalImage, const std::string& watermarkText) {
    if (originalImage.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    if (originalImage.channels() != 1) {
         // ���߳���ת��Ϊ�Ҷ�ͼ
         throw std::invalid_argument("Input image must be grayscale.");
    }
    if (watermarkText.empty()) {
        throw std::invalid_argument("Watermark text cannot be empty.");
    }

    std::cout << "Starting watermark embedding..." << std::endl;

    // Step 1: ��Ե���
    std::cout << "Step 1: Detecting edges..." << std::endl;
    cv::Mat edgeImage = edgeDetector.detectEdges(originalImage);
    std::cout << "Edge detection complete." << std::endl;

    // Step 3: ˮӡ���� (��ǰ��ȡ����)
    std::cout << "Step 3: Encoding watermark..." << std::endl;
    std::vector<int> watermarkBits = watermarkEncoder.encodeWatermark(watermarkText);
    int watermarkLength = watermarkBits.size();
    std::cout << "Watermark encoded into " << watermarkLength << " bits." << std::endl;
    if (watermarkLength <= 0) {
        throw std::runtime_error("Encoded watermark has zero length.");
    }

    // Step 2: ѡ��Ƕ������ (ʹ��ˮӡ������ΪĿ������)
    std::cout << "Step 2: Selecting embedding regions..." << std::endl;
    // ʹ��ԭʼ regionSelector ʵ���� getter ��ȡ����
    RegionSelector regionSelectorForEmbedding(regionScorer, watermarkLength,
                                              regionSelector.getWindowScale(), // ʹ�� getter
                                              regionSelector.getStepScale());  // ʹ�� getter
    std::vector<Region> selectedRegions = regionSelectorForEmbedding.selectEmbeddingRegions(originalImage, edgeImage);

    if (selectedRegions.empty()) {
        throw std::runtime_error("Failed to select any embedding regions.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions for embedding." << std::endl;

    //for (auto& i : selectedRegions) {
    //    std::cout << i.bounds.x << " " << i.bounds.y << std::endl;
    //}

    // 3. ��������������������
     if (selectedRegions.size() < watermarkLength) {
         std::cerr << "Warning: Not enough non-overlapping regions found (" << selectedRegions.size()
                   << ") for the watermark length (" << watermarkLength
                   << "). Embedding will use fewer bits." << std::endl;
         watermarkLength = selectedRegions.size(); // ��ʹ�ÿ��õ�����
         if (watermarkLength == 0) throw std::runtime_error("No regions available to embed watermark.");
         watermarkBits.resize(watermarkLength); // �ض�ˮӡ
     }
     // ���� selectedRegions.size() == watermarkLength


    cv::Mat watermarkedImage = originalImage.clone();
    watermarkedImage.convertTo(watermarkedImage, CV_64F); // ת��ΪCV_64F�����޸�

    std::cout << "Step 4, 5, 6: Processing blocks and embedding bits..." << std::endl;
    for (int i = 0; i < watermarkLength; ++i) {
        const Region& region = selectedRegions[i];
        int watermarkBit = watermarkBits[i];

        // ��ȡ��ǰ�����Ӧ��ͼ���ͱ�Ե��
        cv::Mat regionPatch = originalImage(region.bounds);
        cv::Mat regionEdgePatch = edgeImage(region.bounds);

        // Step 4: ʹ���µĹ����������������
        ImageBlock block = blockProcessor.processRegionAsBlock(regionPatch, regionEdgePatch, region.bounds);

        // Step 5: ���������޸��� w_xy(i,j)
        // block �ṹ���ڰ���������� embeddingStrength �� modificationWeights
        cv::Mat modificationMatrix = blockProcessor.calculatePixelModifications(block, regionPatch, watermarkBit); // ���� CV_64F

        // Step 6: Ӧ���޸� (ʽ 18)
        cv::Mat targetPatch = watermarkedImage(block.bounds); // ��ȡ���޸���������� (CV_64F)
        cv::Mat modificationMatrixFloat;
        modificationMatrix.convertTo(modificationMatrixFloat, CV_64F); // ת��Ϊ CV_64F �Ա����

        cv::add(targetPatch, modificationMatrixFloat, targetPatch); // ֱ���� watermarkedImage ���޸�

        // ������������ӽ���ָʾ
        if ((i + 1) % 10 == 0 || i == watermarkLength - 1) {
             std::cout << "Embedded bit " << (i + 1) << "/" << watermarkLength << std::endl;
        }
    }

    // ��ͼ��ת���� CV_8U �����нض�
    cv::Mat finalWatermarkedImage;
    watermarkedImage.convertTo(finalWatermarkedImage, CV_8U); // ���Զ��ضϵ� [0, 255]

    std::cout << "Watermark embedding complete." << std::endl;
    return finalWatermarkedImage;
}
