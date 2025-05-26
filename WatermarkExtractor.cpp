#include "WatermarkExtractor.h"
#include <stdexcept>
#include <iostream>
#include <cmath>

WatermarkExtractor::WatermarkExtractor(int expectedWatermarkLength, int edgeThreshold)
    : edgeDetector(),
      regionScorer(),
      regionSelector(regionScorer, expectedWatermarkLength),
      blockProcessor(edgeThreshold),
      watermarkDecoder(),
      expectedWatermarkLength(expectedWatermarkLength)
{
    if (expectedWatermarkLength <= 0) {
        throw std::invalid_argument("Expected watermark length must be positive.");
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

    // Step 1: 区域得分，选出4个最高得分区域
    std::cout << "Step 1: Detecting edges and selecting regions..." << std::endl;
    cv::Mat edgeImage = edgeDetector.detectEdges(watermarkedImage);
    RegionSelector regionSelectorForExtraction(regionScorer, 4, regionSelector.getWindowScale(), regionSelector.getStepScale());
    std::vector<Region> selectedRegions = regionSelectorForExtraction.selectEmbeddingRegions(watermarkedImage, edgeImage);
    if (selectedRegions.size() < 4) {
        throw std::runtime_error("Failed to select 4 regions for extraction.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions." << std::endl;

    // Step 2: 对每个区域都完整提取水印（区域分成m块，每块提取1位）
    std::vector<std::vector<int>> allExtractedBits(4);
    cv::Mat watermarkedImageFloat;
    watermarkedImage.convertTo(watermarkedImageFloat, CV_64F);

    for (int regionIdx = 0; regionIdx < 4; ++regionIdx) {
        const Region& region = selectedRegions[regionIdx];
        cv::Mat regionPatch = watermarkedImage(region.bounds);
        cv::Mat regionEdgePatch = edgeImage(region.bounds);

        // 区域分成m块
        int m = expectedWatermarkLength;
        std::vector<ImageBlock> blocks = blockProcessor.prepareBlocks(regionPatch, regionEdgePatch, m);

        std::vector<int> extractedBits;
        extractedBits.reserve(m);

        for (int i = 0; i < m; ++i) {
            const ImageBlock& block = blocks[i];
            int blockWidth = block.bounds.width;
            int blockHeight = block.bounds.height;
            double sigma_xy = block.embeddingStrength;

            if (sigma_xy <= 0 || blockWidth <= 0 || blockHeight <= 0) {
                extractedBits.push_back(0);
                continue;
            }

            cv::Mat blockPatchFloat = watermarkedImageFloat(region.bounds)(block.bounds);
            double dcCoefficient = blockProcessor.calculateDCCoefficient(blockPatchFloat);

            double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
            double quantizationStep = sigma_xy * ab_sqrt;

            if (quantizationStep < 1e-9) {
                extractedBits.push_back(0);
                continue;
            }

            double normalizedDC = dcCoefficient / quantizationStep;
            int floorValue = static_cast<int>(std::floor(normalizedDC));
            int extractedBit = std::abs(floorValue % 2);

            extractedBits.push_back(extractedBit);
        }
        allExtractedBits[regionIdx] = extractedBits;
    }

    // Step 3: 对4个区域的提取结果做投票（多数表决），得到最终比特流
    std::vector<int> finalBits(expectedWatermarkLength, 0);
    for (int i = 0; i < expectedWatermarkLength; ++i) {
        int sum = 0;
        for (int r = 0; r < 4; ++r) {
            sum += allExtractedBits[r][i];
        }
        finalBits[i] = (sum >= 2) ? 1 : 0; // 多数投票
    }

    std::cout << "Extraction of " << finalBits.size() << " bits complete." << std::endl;

    std::cout << "Step 4: Decoding extracted bits..." << std::endl;
    std::string decodedWatermark;
    try {
        decodedWatermark = watermarkDecoder.decodeWatermark(finalBits);
        std::cout << "Decoding complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during decoding: " << e.what() << std::endl;
        return "";
    }

    return decodedWatermark;
}
