#include "WatermarkEncoder.h"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

// Schifra 头文件
#include "schifra_galois_field.hpp"
#include "schifra_galois_field_polynomial.hpp"
#include "schifra_sequential_root_generator_polynomial_creator.hpp"
#include "schifra_reed_solomon_encoder.hpp"
#include "schifra_reed_solomon_block.hpp"

WatermarkEncoder::WatermarkEncoder(int rsN, int rsK, int markerLength)
    : rs_n(rsN), rs_k(rsK), marker_len(markerLength) {
    if (marker_len < 0) {
        throw std::invalid_argument("Marker length cannot be negative.");
    }
    // 在实际应用中，这里会初始化 RS 编码器库
}

// 将字符串转换为二进制位序列 (每个 char 转 8 bits)
std::vector<int> WatermarkEncoder::stringToBits(const std::string& text) {
    std::vector<int> bits;
    bits.reserve(text.length() * 8);
    for (char c : text) {
        for (int i = 7; i >= 0; --i) {
            bits.push_back((c >> i) & 1);
        }
    }
    return bits;
}

// 添加固定的标记位 (这里用全 1 示例)
std::vector<int> WatermarkEncoder::addMarkerBits(const std::vector<int>& data) {
    std::vector<int> markedData;
    markedData.reserve(marker_len + data.size());
    markedData.insert(markedData.end(), data.begin(), data.end());
    // 添加标记位 (例如，41 个 1)
    int size = markedData.size();
    for (int i = markedData.size() - marker_len; i < size; ++i) {
        markedData.push_back(1); // 或者使用更复杂的标记序列
    }
    return markedData;
}

// 使用 Schifra 库实现 RS 编码
std::string WatermarkEncoder::performRSEncoding(const std::string& data) {
    // Schifra模板参数必须为编译期常量
    constexpr std::size_t field_descriptor = 8;
    constexpr std::size_t generator_polynomial_index = 120;
    constexpr std::size_t generator_polynomial_root_count = 32;

    constexpr std::size_t code_length = 255;
    constexpr std::size_t fec_length = 32;
    constexpr std::size_t data_length = 223;

    // 1. 构造伽罗瓦域
    const schifra::galois::field field(
        field_descriptor,
        schifra::galois::primitive_polynomial_size06,
        schifra::galois::primitive_polynomial06
    );

    // 2. 构造生成多项式
    schifra::galois::field_polynomial generator_polynomial(field);
    if (!schifra::make_sequential_root_generator_polynomial(
            field,
            generator_polynomial_index,
            generator_polynomial_root_count,
            generator_polynomial)) {
        std::cerr << "Error - Failed to create sequential root generator!" << std::endl;
        std::cerr << "return raw data" << std::endl;
        return data;
    }

    // 3. 构造编码器
    typedef schifra::reed_solomon::encoder<code_length, fec_length, data_length> encoder_t;
    encoder_t encoder(field, generator_polynomial);

    std::string message(data);
    message.resize(code_length, 0x00); // 填充到code_length

    // 5. 创建RS块
    schifra::reed_solomon::block<code_length, fec_length> block;

    // 6. 编码
    if (!encoder.encode(message, block)) {
        std::cerr << "Error - Critical encoding failure!" << std::endl;
        return data;
    }

    std::ostringstream oss;
    oss << block;
    std::string codeword = oss.str();

    return codeword.substr(0, 8)+ codeword.substr(codeword.size() - 32, 32);
}

std::vector<int> stringToBitStream(const std::string& str) {
    std::vector<int> bits;
    bits.reserve(str.size() * 8);
    for (unsigned char ch : str) {
        // 从最高位到最低位依次提取
        for (int b = 7; b >= 0; --b) {
            bits.push_back((ch >> b) & 0x01);
        }
    }
    return bits;
}

// 编码水印 (对应 Step 3)
std::vector<int> WatermarkEncoder::encodeWatermark(const std::string& originalWatermark) {
    if (originalWatermark.empty()) {
        throw std::invalid_argument("Original watermark text cannot be empty.");
    }

    // 1. (占位) RS 编码
    std::string encodedWatermark = performRSEncoding(originalWatermark);

    std::vector<int> encodedBits = stringToBits(encodedWatermark);

    // 2. 添加标记信息
    std::vector<int> finalBits = addMarkerBits(encodedBits);

    /*for (auto& i : finalBits) {
        std::cout << i;
    }
    std::cout << std::endl;*/

    // 检查最终长度是否合理 (例如，不能为 0)
    if (finalBits.empty()) {
         throw std::runtime_error("Watermark encoding resulted in an empty bit sequence.");
    }

    return finalBits;
}
