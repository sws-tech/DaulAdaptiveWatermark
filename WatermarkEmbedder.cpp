#include "WatermarkEmbedder.h"
#include <stdexcept>
#include <iostream>
#include "utils.h"

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
        throw std::invalid_argument("Input image must be single channel (Y channel).");
    }
    if (watermarkText.empty()) {
        throw std::invalid_argument("Watermark text cannot be empty.");
    }

    std::cout << "Starting watermark embedding..." << std::endl;

    // Step 1: ��Ե���
    std::cout << "Step 1: Detecting edges..." << std::endl;
    cv::Mat edgeImage = edgeDetector.detectEdges(originalImage);
    std::cout << "Edge detection complete." << std::endl;

    // Step 2: ����÷֣�ѡ��4����ߵ÷�����
    std::cout << "Step 2: Selecting top 4 embedding regions..." << std::endl;
    RegionSelector regionSelectorForEmbedding(regionScorer, 4, regionSelector.getWindowScale(), regionSelector.getStepScale());
    std::vector<Region> selectedRegions = regionSelectorForEmbedding.selectEmbeddingRegions(originalImage, edgeImage);
    if (selectedRegions.size() < 4) {
        throw std::runtime_error("Failed to select 4 embedding regions.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions for embedding." << std::endl;

    // Step 3: ˮӡ����
    std::cout << "Step 3: Encoding watermark..." << std::endl;
    std::vector<int> watermarkBits = watermarkEncoder.encodeWatermark(watermarkText);
    int watermarkLength = watermarkBits.size();
    std::cout << "Watermark encoded into " << watermarkLength << " bits." << std::endl;
    if (watermarkLength <= 0) {
        throw std::runtime_error("Encoded watermark has zero length.");
    }

    // Step 4: ��ÿ����������Ƕ������ˮӡ������ֳ�m�飬ÿ��Ƕ��1λ��
    cv::Mat watermarkedImage = originalImage.clone();
    watermarkedImage.convertTo(watermarkedImage, CV_64F);

    for (int regionIdx = 0; regionIdx < 4; ++regionIdx) {
        const Region& region = selectedRegions[regionIdx];
        cv::Mat regionPatch = originalImage(region.bounds);
        cv::Mat regionEdgePatch = edgeImage(region.bounds);

        // ������ֳ�m��
        std::vector<ImageBlock> blocks = blockProcessor.prepareBlocks(regionPatch, regionEdgePatch, watermarkLength);

        for (int i = 0; i < watermarkLength; ++i) {
            const ImageBlock& block = blocks[i];
            int watermarkBit = watermarkBits[i];

            cv::Mat blockPatch = regionPatch(block.bounds);
            cv::Mat modificationMatrix = blockProcessor.calculatePixelModifications(block, blockPatch, watermarkBit);

            // Ӧ���޸�
            cv::Mat targetPatch = watermarkedImage(region.bounds)(block.bounds);
            cv::Mat modificationMatrixFloat;
            modificationMatrix.convertTo(modificationMatrixFloat, CV_64F);
            cv::add(targetPatch, modificationMatrixFloat, targetPatch);
        }
    }

    // ת��8λ
    cv::Mat finalWatermarkedImage;
    watermarkedImage.convertTo(finalWatermarkedImage, CV_8U);

    std::cout << "Watermark embedding complete." << std::endl;
    return finalWatermarkedImage;
}
