#include "WatermarkEmbedder.h"
#include <stdexcept>
#include <iostream> // For warnings/errors
#include "utils.h" // 需要 calculateGaussianWeights (现在不需要直接调用了)

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
         // 或者尝试转换为灰度图
         throw std::invalid_argument("Input image must be grayscale.");
    }
    if (watermarkText.empty()) {
        throw std::invalid_argument("Watermark text cannot be empty.");
    }

    std::cout << "Starting watermark embedding..." << std::endl;

    // Step 1: 边缘检测
    std::cout << "Step 1: Detecting edges..." << std::endl;
    cv::Mat edgeImage = edgeDetector.detectEdges(originalImage);
    std::cout << "Edge detection complete." << std::endl;

    // Step 3: 水印编码 (提前获取长度)
    std::cout << "Step 3: Encoding watermark..." << std::endl;
    std::vector<int> watermarkBits = watermarkEncoder.encodeWatermark(watermarkText);
    int watermarkLength = watermarkBits.size();
    std::cout << "Watermark encoded into " << watermarkLength << " bits." << std::endl;
    if (watermarkLength <= 0) {
        throw std::runtime_error("Encoded watermark has zero length.");
    }

    // Step 2: 选择嵌入区域 (使用水印长度作为目标数量)
    std::cout << "Step 2: Selecting embedding regions..." << std::endl;
    // 使用原始 regionSelector 实例的 getter 获取参数
    RegionSelector regionSelectorForEmbedding(regionScorer, watermarkLength,
                                              regionSelector.getWindowScale(), // 使用 getter
                                              regionSelector.getStepScale());  // 使用 getter
    std::vector<Region> selectedRegions = regionSelectorForEmbedding.selectEmbeddingRegions(originalImage, edgeImage);

    if (selectedRegions.empty()) {
        throw std::runtime_error("Failed to select any embedding regions.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions for embedding." << std::endl;

    //for (auto& i : selectedRegions) {
    //    std::cout << i.bounds.x << " " << i.bounds.y << std::endl;
    //}

    // 3. 处理区域数量不足的情况
     if (selectedRegions.size() < watermarkLength) {
         std::cerr << "Warning: Not enough non-overlapping regions found (" << selectedRegions.size()
                   << ") for the watermark length (" << watermarkLength
                   << "). Embedding will use fewer bits." << std::endl;
         watermarkLength = selectedRegions.size(); // 仅使用可用的区域
         if (watermarkLength == 0) throw std::runtime_error("No regions available to embed watermark.");
         watermarkBits.resize(watermarkLength); // 截断水印
     }
     // 现在 selectedRegions.size() == watermarkLength


    cv::Mat watermarkedImage = originalImage.clone();
    watermarkedImage.convertTo(watermarkedImage, CV_64F); // 转换为CV_64F进行修改

    std::cout << "Step 4, 5, 6: Processing blocks and embedding bits..." << std::endl;
    for (int i = 0; i < watermarkLength; ++i) {
        const Region& region = selectedRegions[i];
        int watermarkBit = watermarkBits[i];

        // 获取当前区域对应的图像块和边缘块
        cv::Mat regionPatch = originalImage(region.bounds);
        cv::Mat regionEdgePatch = edgeImage(region.bounds);

        // Step 4: 使用新的公共方法计算块属性
        ImageBlock block = blockProcessor.processRegionAsBlock(regionPatch, regionEdgePatch, region.bounds);

        // Step 5: 计算像素修改量 w_xy(i,j)
        // block 结构现在包含了所需的 embeddingStrength 和 modificationWeights
        cv::Mat modificationMatrix = blockProcessor.calculatePixelModifications(block, regionPatch, watermarkBit); // 返回 CV_64F

        // Step 6: 应用修改 (式 18)
        cv::Mat targetPatch = watermarkedImage(block.bounds); // 获取待修改区域的引用 (CV_64F)
        cv::Mat modificationMatrixFloat;
        modificationMatrix.convertTo(modificationMatrixFloat, CV_64F); // 转换为 CV_64F 以便相加

        cv::add(targetPatch, modificationMatrixFloat, targetPatch); // 直接在 watermarkedImage 上修改

        // 可以在这里添加进度指示
        if ((i + 1) % 10 == 0 || i == watermarkLength - 1) {
             std::cout << "Embedded bit " << (i + 1) << "/" << watermarkLength << std::endl;
        }
    }

    // 将图像转换回 CV_8U 并进行截断
    cv::Mat finalWatermarkedImage;
    watermarkedImage.convertTo(finalWatermarkedImage, CV_8U); // 会自动截断到 [0, 255]

    std::cout << "Watermark embedding complete." << std::endl;
    return finalWatermarkedImage;
}
