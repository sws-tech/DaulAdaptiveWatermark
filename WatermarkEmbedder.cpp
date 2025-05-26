#include "WatermarkEmbedder.h"
#include <stdexcept>
#include <iostream>
#include "utils.h"

WatermarkEmbedder::WatermarkEmbedder(int numRegions, int edgeThreshold)
    : edgeDetector(), // 使用默认参数或传入特定参数
      regionScorer(), // 使用默认权重或传入特定权重
      regionSelector(regionScorer, numRegions), // 传入 scorer 和目标区域数
      watermarkEncoder(), // 使用默认参数
      blockProcessor(edgeThreshold), // 传入边缘块阈值 Th
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

    // Step 1: 边缘检测
    std::cout << "Step 1: Detecting edges..." << std::endl;
    cv::Mat edgeImage = edgeDetector.detectEdges(originalImage);
    std::cout << "Edge detection complete." << std::endl;

    // Step 2: 区域得分，选出4个最高得分区域
    std::cout << "Step 2: Selecting top 4 embedding regions..." << std::endl;
    RegionSelector regionSelectorForEmbedding(regionScorer, 4, regionSelector.getWindowScale(), regionSelector.getStepScale());
    std::vector<Region> selectedRegions = regionSelectorForEmbedding.selectEmbeddingRegions(originalImage, edgeImage);
    if (selectedRegions.size() < 4) {
        throw std::runtime_error("Failed to select 4 embedding regions.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions for embedding." << std::endl;

    // Step 3: 水印编码
    std::cout << "Step 3: Encoding watermark..." << std::endl;
    std::vector<int> watermarkBits = watermarkEncoder.encodeWatermark(watermarkText);
    int watermarkLength = watermarkBits.size();
    std::cout << "Watermark encoded into " << watermarkLength << " bits." << std::endl;
    if (watermarkLength <= 0) {
        throw std::runtime_error("Encoded watermark has zero length.");
    }

    // Step 4: 对每个区域都完整嵌入整个水印（区域分成m块，每块嵌入1位）
    cv::Mat watermarkedImage = originalImage.clone();
    watermarkedImage.convertTo(watermarkedImage, CV_64F);

    for (int regionIdx = 0; regionIdx < 4; ++regionIdx) {
        const Region& region = selectedRegions[regionIdx];
        cv::Mat regionPatch = originalImage(region.bounds);
        cv::Mat regionEdgePatch = edgeImage(region.bounds);

        // 将区域分成m块
        std::vector<ImageBlock> blocks = blockProcessor.prepareBlocks(regionPatch, regionEdgePatch, watermarkLength);

        for (int i = 0; i < watermarkLength; ++i) {
            const ImageBlock& block = blocks[i];
            int watermarkBit = watermarkBits[i];

            cv::Mat blockPatch = regionPatch(block.bounds);
            cv::Mat modificationMatrix = blockProcessor.calculatePixelModifications(block, blockPatch, watermarkBit);

            // 应用修改
            cv::Mat targetPatch = watermarkedImage(region.bounds)(block.bounds);
            cv::Mat modificationMatrixFloat;
            modificationMatrix.convertTo(modificationMatrixFloat, CV_64F);
            cv::add(targetPatch, modificationMatrixFloat, targetPatch);
        }
    }

    // 转回8位
    cv::Mat finalWatermarkedImage;
    watermarkedImage.convertTo(finalWatermarkedImage, CV_8U);

    std::cout << "Watermark embedding complete." << std::endl;
    return finalWatermarkedImage;
}
