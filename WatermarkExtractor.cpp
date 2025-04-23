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
        throw std::invalid_argument("Input watermarked image must be grayscale.");
    }

    std::cout << "Starting watermark extraction..." << std::endl;

    // Step 1: 检测边缘，定位嵌入区域 (与嵌入过程一致)
    std::cout << "Step 1: Detecting edges and selecting regions..." << std::endl;
    //cv::Mat edgeImage = edgeDetector.detectEdges(watermarkedImage); // 在含水印图像上检测
    //cv::imwrite("de_edge.png", edgeImage);
    // 注意：理论上应该在原始图像上检测边缘和选择区域，但提取时通常只有含水印图像
    // 这里假设攻击对边缘和区域选择的影响不大，或者嵌入/提取使用相同的图像版本
    cv::Mat raw_image = cv::imread(rawImagePath, cv::IMREAD_GRAYSCALE);

    if (raw_image.empty()) {
        std::cerr << "Failed to load image: " << rawImagePath << std::endl;
    }
    cv::Mat edgeImage = edgeDetector.detectEdges(raw_image); // 在含水印图像上检测

    // 使用预期的水印长度 m 作为目标区域数
    RegionSelector regionSelectorForExtraction(regionScorer, expectedWatermarkLength,
                                               regionSelector.getWindowScale(),
                                               regionSelector.getStepScale());
    std::vector<Region> selectedRegions = regionSelectorForExtraction.selectEmbeddingRegions(raw_image, edgeImage); // 在含水印图像上选择

    if (selectedRegions.empty()) {
        throw std::runtime_error("Failed to select any regions for extraction.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions." << std::endl;

    //for (auto& i : selectedRegions) {
    //    std::cout << i.bounds.x << " " << i.bounds.y << std::endl;
    //}

    // 处理区域数量不足的情况
    int numRegionsFound = selectedRegions.size();
    if (numRegionsFound < expectedWatermarkLength) {
        std::cerr << "Warning: Found only " << numRegionsFound << " regions, expected " << expectedWatermarkLength << ". Extraction might be incomplete." << std::endl;
        // 可以选择继续提取找到的部分，或者报错
    }
    int bitsToExtract = std::min(numRegionsFound, expectedWatermarkLength);
    if (bitsToExtract == 0) {
         throw std::runtime_error("No regions available to extract watermark bits.");
    }


    std::vector<int> extractedBits;
    extractedBits.reserve(bitsToExtract);

    cv::Mat watermarkedImageFloat;
    watermarkedImage.convertTo(watermarkedImageFloat, CV_64F); // 转换为CV_64F进行DC计算

    std::cout << "Step 2 & 3: Processing blocks and extracting bits..." << std::endl;
    for (int i = 0; i < bitsToExtract; ++i) {
        const Region& region = selectedRegions[i];

        // 获取当前区域对应的图像块和边缘块
        cv::Mat regionPatch = watermarkedImage(region.bounds); // 从含水印图像提取
        cv::Mat regionEdgePatch = edgeImage(region.bounds); // 从计算出的边缘图提取

        // Step 2: 计算块参数 (特别是 sigma_xy)
        // 使用 BlockProcessor 的公共方法来获取块信息
        ImageBlock block = blockProcessor.processRegionAsBlock(regionPatch, regionEdgePatch, region.bounds);
        double sigma_xy = block.embeddingStrength; // 获取计算出的嵌入强度
        int blockWidth = block.bounds.width;
        int blockHeight = block.bounds.height;

        if (sigma_xy <= 0 || blockWidth <= 0 || blockHeight <= 0) {
             std::cerr << "Warning: Invalid block parameters for region " << i << ". Skipping bit extraction." << std::endl;
             // 可以插入一个默认值 (如 0) 或标记错误
             extractedBits.push_back(0); // 插入默认值
             continue;
        }

        // Step 3: 提取 DC 系数并提取水印位 (式 19)
        cv::Mat blockPatchFloat = watermarkedImageFloat(region.bounds); // 从浮点图像提取
        double dcCoefficient = blockProcessor.calculateDCCoefficient(blockPatchFloat); // R'_DCxy

        double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
        double quantizationStep = sigma_xy * ab_sqrt;

        if (quantizationStep < 1e-9) {
            std::cerr << "Warning: Quantization step too small for region " << i << ". Skipping bit extraction." << std::endl;
            extractedBits.push_back(0); // 插入默认值
            continue;
        }

        // 计算 floor(R'_DC / step) mod 2
        double normalizedDC = dcCoefficient / quantizationStep;
        int floorValue = static_cast<int>(std::floor(normalizedDC));
        int extractedBit = std::abs(floorValue % 2); // 使用 abs 确保结果是 0 或 1

        // --- 另一种 QIM 提取逻辑 (对应之前的嵌入逻辑) ---
        // double roundedValue = std::round(normalizedDC);
        // double diff = normalizedDC - roundedValue; // 判断在整数的左边还是右边
        // if (std::abs(diff) < 0.25) { // 靠近整数，认为是 0
        //     extractedBit = 0;
        // } else { // 靠近半整数，认为是 1
        //     extractedBit = 1;
        // }
        // --- 结束 ---
        // 我们将使用式 (19) 的直接实现

        extractedBits.push_back(extractedBit);

        if ((i + 1) % 10 == 0 || i == bitsToExtract - 1) {
             std::cout << "Extracted bit " << (i + 1) << "/" << bitsToExtract << std::endl;
        }
    }

    std::cout << "Extraction of " << extractedBits.size() << " bits complete." << std::endl;

    // Step 4: 解码
    std::cout << "Step 4: Decoding extracted bits..." << std::endl;
    std::string decodedWatermark;
    try {
        decodedWatermark = watermarkDecoder.decodeWatermark(extractedBits);
        std::cout << "Decoding complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during decoding: " << e.what() << std::endl;
        // 根据需要决定是返回空字符串还是重新抛出异常
        return ""; // 返回空字符串表示解码失败
    }

    return decodedWatermark;
}

// 需要 BlockProcessor::calculateDCCoefficient 成为 public 或提供一个包装器
// 检查 BlockProcessor.h 和 .cpp
// 确认 calculateDCCoefficient 是否已经是 public 或需要调整
