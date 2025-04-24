#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "WatermarkEmbedder.h"
#include "WatermarkExtractor.h" // 包含提取器头文件

// 函数：打印用法说明
void printUsage(const char* progName) {
    std::cerr << "Usage: " << std::endl;
    std::cerr << "  " << progName << " embed <input_image> <output_image> <watermark_text> [num_regions] [edge_threshold]" << std::endl;
    std::cerr << "  " << progName << " extract <input_image> <expected_watermark_length> [edge_threshold]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  embed:        Embed a watermark." << std::endl;
    std::cerr << "  extract:      Extract a watermark." << std::endl;
    std::cerr << "  <input_image>: Path to the input image (grayscale)." << std::endl;
    std::cerr << "  <output_image>: Path to save the watermarked image (embed mode only)." << std::endl;
    std::cerr << "  <watermark_text>: The text to embed (embed mode only)." << std::endl;
    std::cerr << "  <expected_watermark_length>: The expected length of the encoded watermark in bits (extract mode only)." << std::endl;
    std::cerr << "  [num_regions]: (Optional, embed mode) Number of regions to select (default: derived from watermark length)." << std::endl;
    std::cerr << "  [edge_threshold]: (Optional) Threshold for classifying edge blocks (default: 5)." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example (Embed):" << std::endl;
    std::cerr << "  " << progName << " embed input.png output.png \"Secret Message\" 100 10" << std::endl;
    std::cerr << "Example (Extract):" << std::endl;
    std::cerr << "  " << progName << " extract output.png 100 10" << std::endl; // Assuming encoded length is 100 bits
}


int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    if (argc < 3) {
        printUsage(argv[0]);
        return -1;
    }

    std::string mode = argv[1];
    std::string inputImagePath = argv[2];

    // 默认参数
    int edgeThreshold = 5; // 默认边缘块阈值

    try {
        // 加载输入图像 (彩色)
        cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load image: " << inputImagePath << std::endl;
            return -1;
        }

        if (mode == "embed") {
            if (argc < 5) {
                std::cerr << "Error: Missing arguments for embed mode." << std::endl;
                printUsage(argv[0]);
                return -1;
            }
            std::string outputImagePath = argv[3];
            std::string watermarkText = argv[4];
            int numRegions = 0; // 0 表示由水印长度决定

            // 限制水印最大长度为8
            if (watermarkText.length() > 8) {
                watermarkText = watermarkText.substr(0, 8);
                std::cout << "Watermark text truncated to 8 characters: " << watermarkText << std::endl;
            }

            // 解析可选参数
            if (argc > 5) {
                try {
                    numRegions = std::stoi(argv[5]); // 用户指定区域数 (可能覆盖水印长度决定)
                    if (numRegions <= 0) {
                         std::cerr << "Warning: num_regions must be positive. Using default logic." << std::endl;
                         numRegions = 0; // 回退到默认逻辑
                    }
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid num_regions value. Using default logic." << std::endl;
                     numRegions = 0;
                } catch (const std::out_of_range& e) {
                     std::cerr << "Warning: num_regions value out of range. Using default logic." << std::endl;
                     numRegions = 0;
                }
            }
             if (argc > 6) {
                 try {
                     edgeThreshold = std::stoi(argv[6]);
                 } catch (const std::exception& e) {
                     std::cerr << "Warning: Invalid edge_threshold value. Using default: " << edgeThreshold << std::endl;
                 }
             }

            // 转为YUV用于水印处理
            cv::Mat yuvInput;
            cv::cvtColor(inputImage, yuvInput, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> yuvChannels;
            cv::split(yuvInput, yuvChannels);

            // 创建嵌入器实例
            // 如果 numRegions 为 0，WatermarkEmbedder 内部会根据水印长度确定区域数
            WatermarkEmbedder embedder(numRegions == 0 ? 10 : numRegions, edgeThreshold); // 提供一个默认值以防万一

            // 执行水印嵌入（在Y通道上）
            std::cout << "Embedding watermark..." << std::endl;
            cv::Mat watermarkedY = embedder.embedWatermark(yuvChannels[0], watermarkText);

            // 替换Y通道
            yuvChannels[0] = watermarkedY;
            cv::Mat watermarkedYUV, watermarkedBGR;
            cv::merge(yuvChannels, watermarkedYUV);
            cv::cvtColor(watermarkedYUV, watermarkedBGR, cv::COLOR_YCrCb2BGR);

            // 保存含水印图像
            if (cv::imwrite(outputImagePath, watermarkedBGR)) {
                std::cout << "Watermark embedded successfully. Output saved to: " << outputImagePath << std::endl;
            } else {
                std::cerr << "Error: Could not save watermarked image to: " << outputImagePath << std::endl;
                return -1;
            }

        } else if (mode == "extract") {
            // 默认水印长度为361位，不再通过参数传入
            int expectedLength = 361;

            // 解析可选参数
            if (argc > 3) {
                try {
                    edgeThreshold = std::stoi(argv[3]);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Invalid edge_threshold value. Using default: " << edgeThreshold << std::endl;
                }
            }

            // 提取时也从Y通道
            cv::Mat yuvInput;
            cv::cvtColor(inputImage, yuvInput, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> yuvChannels;
            cv::split(yuvInput, yuvChannels);

            // 创建提取器实例（不再需要原始图像路径）
            WatermarkExtractor extractor(expectedLength, edgeThreshold);

            // 执行水印提取
            std::cout << "Extracting watermark..." << std::endl;
            std::string extractedText = extractor.extractWatermark(yuvChannels[0]);

            if (!extractedText.empty()) {
                std::cout << "Watermark extracted successfully:" << std::endl;
                std::cout << extractedText << std::endl;
            } else {
                std::cout << "Watermark extraction failed or resulted in empty text." << std::endl;
                return 1;
            }

        } else {
            std::cerr << "Error: Invalid mode specified. Use 'embed' or 'extract'." << std::endl;
            printUsage(argv[0]);
            return -1;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
