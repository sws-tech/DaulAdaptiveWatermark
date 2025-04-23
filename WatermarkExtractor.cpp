#include "WatermarkExtractor.h"
#include <stdexcept>
#include <iostream>
#include <cmath> // for std::floor

WatermarkExtractor::WatermarkExtractor(int expectedWatermarkLength, const std::string& rawImagePath, int edgeThreshold)
    : edgeDetector(),
      regionScorer(),
      // 初始化 RegionSelector 时，目标区域数应为预期的水印长度 m (基于简化实现)
      regionSelector(regionScorer, expectedWatermarkLength),
      blockProcessor(edgeThreshold),
      watermarkDecoder(), // 使用默认解码器参数
      expectedWatermarkLength(expectedWatermarkLength),
      rawImagePath(rawImagePath) // 保存路径
{
    if (expectedWatermarkLength <= 0) {
        throw std::invalid_argument("Expected watermark length must be positive.");
    }
    if (rawImagePath.empty()) {
        throw std::invalid_argument("Raw image path must not be empty.");
    }
}

std::string WatermarkExtractor::extractWatermark(const cv::Mat& watermarkedImage) {
    if (watermarkedImage.empty()) {
        throw std::invalid_argument("Input watermarked image is empty.");
    }
    if (watermarkedImage.channels() != 1) {
        throw std::invalid_argument("Input watermarked image must be single channel (Y channel).");
    }

    std::cout << "Starting watermark extraction..." << std::endl;

    // Step 1: 检测边缘，定位嵌入区域 (与嵌入过程一致)
    std::cout << "Step 1: Detecting edges and selecting regions..." << std::endl;
    cv::Mat raw_image = cv::imread(rawImagePath, cv::IMREAD_COLOR);
    cv::Mat raw_yuv, raw_y;
    if (!raw_image.empty()) {
        cv::cvtColor(raw_image, raw_yuv, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> yuvChannels;
        cv::split(raw_yuv, yuvChannels);
        raw_y = yuvChannels[0];
    } else {
        std::cerr << "Failed to load image: " << rawImagePath << std::endl;
    }
    cv::Mat edgeImage = edgeDetector.detectEdges(raw_y);

    RegionSelector regionSelectorForExtraction(regionScorer, expectedWatermarkLength,
                                               regionSelector.getWindowScale(),
                                               regionSelector.getStepScale());
    std::vector<Region> selectedRegions = regionSelectorForExtraction.selectEmbeddingRegions(raw_y, edgeImage);

    if (selectedRegions.empty()) {
        throw std::runtime_error("Failed to select any regions for extraction.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions." << std::endl;

    int numRegionsFound = selectedRegions.size();
    if (numRegionsFound < expectedWatermarkLength) {
        std::cerr << "Warning: Found only " << numRegionsFound << " regions, expected " << expectedWatermarkLength << ". Extraction might be incomplete." << std::endl;
    }
    int bitsToExtract = std::min(numRegionsFound, expectedWatermarkLength);
    if (bitsToExtract == 0) {
         throw std::runtime_error("No regions available to extract watermark bits.");
    }

    std::vector<int> extractedBits;
    extractedBits.reserve(bitsToExtract);

    cv::Mat watermarkedImageFloat;
    watermarkedImage.convertTo(watermarkedImageFloat, CV_64F);

    std::cout << "Step 2 & 3: Processing blocks and extracting bits..." << std::endl;
    for (int i = 0; i < bitsToExtract; ++i) {
        const Region& region = selectedRegions[i];

        cv::Mat regionPatch = watermarkedImage(region.bounds);
        cv::Mat regionEdgePatch = edgeImage(region.bounds);

        ImageBlock block = blockProcessor.processRegionAsBlock(regionPatch, regionEdgePatch, region.bounds);
        double sigma_xy = block.embeddingStrength;
        int blockWidth = block.bounds.width;
        int blockHeight = block.bounds.height;

        if (sigma_xy <= 0 || blockWidth <= 0 || blockHeight <= 0) {
             std::cerr << "Warning: Invalid block parameters for region " << i << ". Skipping bit extraction." << std::endl;
             extractedBits.push_back(0);
             continue;
        }

        cv::Mat blockPatchFloat = watermarkedImageFloat(region.bounds);
        double dcCoefficient = blockProcessor.calculateDCCoefficient(blockPatchFloat);

        double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
        double quantizationStep = sigma_xy * ab_sqrt;

        if (quantizationStep < 1e-9) {
            std::cerr << "Warning: Quantization step too small for region " << i << ". Skipping bit extraction." << std::endl;
            extractedBits.push_back(0);
            continue;
        }

        double normalizedDC = dcCoefficient / quantizationStep;
        int floorValue = static_cast<int>(std::floor(normalizedDC));
        int extractedBit = std::abs(floorValue % 2);

        extractedBits.push_back(extractedBit);

        if ((i + 1) % 10 == 0 || i == bitsToExtract - 1) {
             std::cout << "Extracted bit " << (i + 1) << "/" << bitsToExtract << std::endl;
        }
    }

    std::cout << "Extraction of " << extractedBits.size() << " bits complete." << std::endl;

    std::cout << "Step 4: Decoding extracted bits..." << std::endl;
    std::string decodedWatermark;
    try {
        decodedWatermark = watermarkDecoder.decodeWatermark(extractedBits);
        std::cout << "Decoding complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during decoding: " << e.what() << std::endl;
        return "";
    }

    return decodedWatermark;
}

// 需要 BlockProcessor::calculateDCCoefficient 成为 public 或提供一个包装器
// 检查 BlockProcessor.h 和 .cpp
// 确认 calculateDCCoefficient 是否已经是 public 或需要调整
