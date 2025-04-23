#include "WatermarkExtractor.h"
#include <stdexcept>
#include <iostream>
#include <cmath> // for std::floor

WatermarkExtractor::WatermarkExtractor(int expectedWatermarkLength, const std::string& rawImagePath, int edgeThreshold)
    : edgeDetector(),
      regionScorer(),
      // ��ʼ�� RegionSelector ʱ��Ŀ��������ӦΪԤ�ڵ�ˮӡ���� m (���ڼ�ʵ��)
      regionSelector(regionScorer, expectedWatermarkLength),
      blockProcessor(edgeThreshold),
      watermarkDecoder(), // ʹ��Ĭ�Ͻ���������
      expectedWatermarkLength(expectedWatermarkLength),
      rawImagePath(rawImagePath) // ����·��
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

    // Step 1: ����Ե����λǶ������ (��Ƕ�����һ��)
    std::cout << "Step 1: Detecting edges and selecting regions..." << std::endl;
    //cv::Mat edgeImage = edgeDetector.detectEdges(watermarkedImage); // �ں�ˮӡͼ���ϼ��
    //cv::imwrite("de_edge.png", edgeImage);
    // ע�⣺������Ӧ����ԭʼͼ���ϼ���Ե��ѡ�����򣬵���ȡʱͨ��ֻ�к�ˮӡͼ��
    // ������蹥���Ա�Ե������ѡ���Ӱ�첻�󣬻���Ƕ��/��ȡʹ����ͬ��ͼ��汾
    cv::Mat raw_image = cv::imread(rawImagePath, cv::IMREAD_GRAYSCALE);

    if (raw_image.empty()) {
        std::cerr << "Failed to load image: " << rawImagePath << std::endl;
    }
    cv::Mat edgeImage = edgeDetector.detectEdges(raw_image); // �ں�ˮӡͼ���ϼ��

    // ʹ��Ԥ�ڵ�ˮӡ���� m ��ΪĿ��������
    RegionSelector regionSelectorForExtraction(regionScorer, expectedWatermarkLength,
                                               regionSelector.getWindowScale(),
                                               regionSelector.getStepScale());
    std::vector<Region> selectedRegions = regionSelectorForExtraction.selectEmbeddingRegions(raw_image, edgeImage); // �ں�ˮӡͼ����ѡ��

    if (selectedRegions.empty()) {
        throw std::runtime_error("Failed to select any regions for extraction.");
    }
    std::cout << "Selected " << selectedRegions.size() << " regions." << std::endl;

    //for (auto& i : selectedRegions) {
    //    std::cout << i.bounds.x << " " << i.bounds.y << std::endl;
    //}

    // ��������������������
    int numRegionsFound = selectedRegions.size();
    if (numRegionsFound < expectedWatermarkLength) {
        std::cerr << "Warning: Found only " << numRegionsFound << " regions, expected " << expectedWatermarkLength << ". Extraction might be incomplete." << std::endl;
        // ����ѡ�������ȡ�ҵ��Ĳ��֣����߱���
    }
    int bitsToExtract = std::min(numRegionsFound, expectedWatermarkLength);
    if (bitsToExtract == 0) {
         throw std::runtime_error("No regions available to extract watermark bits.");
    }


    std::vector<int> extractedBits;
    extractedBits.reserve(bitsToExtract);

    cv::Mat watermarkedImageFloat;
    watermarkedImage.convertTo(watermarkedImageFloat, CV_64F); // ת��ΪCV_64F����DC����

    std::cout << "Step 2 & 3: Processing blocks and extracting bits..." << std::endl;
    for (int i = 0; i < bitsToExtract; ++i) {
        const Region& region = selectedRegions[i];

        // ��ȡ��ǰ�����Ӧ��ͼ���ͱ�Ե��
        cv::Mat regionPatch = watermarkedImage(region.bounds); // �Ӻ�ˮӡͼ����ȡ
        cv::Mat regionEdgePatch = edgeImage(region.bounds); // �Ӽ�����ı�Եͼ��ȡ

        // Step 2: �������� (�ر��� sigma_xy)
        // ʹ�� BlockProcessor �Ĺ�����������ȡ����Ϣ
        ImageBlock block = blockProcessor.processRegionAsBlock(regionPatch, regionEdgePatch, region.bounds);
        double sigma_xy = block.embeddingStrength; // ��ȡ�������Ƕ��ǿ��
        int blockWidth = block.bounds.width;
        int blockHeight = block.bounds.height;

        if (sigma_xy <= 0 || blockWidth <= 0 || blockHeight <= 0) {
             std::cerr << "Warning: Invalid block parameters for region " << i << ". Skipping bit extraction." << std::endl;
             // ���Բ���һ��Ĭ��ֵ (�� 0) ���Ǵ���
             extractedBits.push_back(0); // ����Ĭ��ֵ
             continue;
        }

        // Step 3: ��ȡ DC ϵ������ȡˮӡλ (ʽ 19)
        cv::Mat blockPatchFloat = watermarkedImageFloat(region.bounds); // �Ӹ���ͼ����ȡ
        double dcCoefficient = blockProcessor.calculateDCCoefficient(blockPatchFloat); // R'_DCxy

        double ab_sqrt = std::sqrt(static_cast<double>(blockWidth * blockHeight));
        double quantizationStep = sigma_xy * ab_sqrt;

        if (quantizationStep < 1e-9) {
            std::cerr << "Warning: Quantization step too small for region " << i << ". Skipping bit extraction." << std::endl;
            extractedBits.push_back(0); // ����Ĭ��ֵ
            continue;
        }

        // ���� floor(R'_DC / step) mod 2
        double normalizedDC = dcCoefficient / quantizationStep;
        int floorValue = static_cast<int>(std::floor(normalizedDC));
        int extractedBit = std::abs(floorValue % 2); // ʹ�� abs ȷ������� 0 �� 1

        // --- ��һ�� QIM ��ȡ�߼� (��Ӧ֮ǰ��Ƕ���߼�) ---
        // double roundedValue = std::round(normalizedDC);
        // double diff = normalizedDC - roundedValue; // �ж�����������߻����ұ�
        // if (std::abs(diff) < 0.25) { // ������������Ϊ�� 0
        //     extractedBit = 0;
        // } else { // ��������������Ϊ�� 1
        //     extractedBit = 1;
        // }
        // --- ���� ---
        // ���ǽ�ʹ��ʽ (19) ��ֱ��ʵ��

        extractedBits.push_back(extractedBit);

        if ((i + 1) % 10 == 0 || i == bitsToExtract - 1) {
             std::cout << "Extracted bit " << (i + 1) << "/" << bitsToExtract << std::endl;
        }
    }

    std::cout << "Extraction of " << extractedBits.size() << " bits complete." << std::endl;

    // Step 4: ����
    std::cout << "Step 4: Decoding extracted bits..." << std::endl;
    std::string decodedWatermark;
    try {
        decodedWatermark = watermarkDecoder.decodeWatermark(extractedBits);
        std::cout << "Decoding complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during decoding: " << e.what() << std::endl;
        // ������Ҫ�����Ƿ��ؿ��ַ������������׳��쳣
        return ""; // ���ؿ��ַ�����ʾ����ʧ��
    }

    return decodedWatermark;
}

// ��Ҫ BlockProcessor::calculateDCCoefficient ��Ϊ public ���ṩһ����װ��
// ��� BlockProcessor.h �� .cpp
// ȷ�� calculateDCCoefficient �Ƿ��Ѿ��� public ����Ҫ����
