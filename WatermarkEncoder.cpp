#include "WatermarkEncoder.h"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

// Schifra ͷ�ļ�
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
    // ��ʵ��Ӧ���У�������ʼ�� RS ��������
}

// ���ַ���ת��Ϊ������λ���� (ÿ�� char ת 8 bits)
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

// ��ӹ̶��ı��λ (������ȫ 1 ʾ��)
std::vector<int> WatermarkEncoder::addMarkerBits(const std::vector<int>& data) {
    std::vector<int> markedData;
    markedData.reserve(marker_len + data.size());
    markedData.insert(markedData.end(), data.begin(), data.end());
    // ��ӱ��λ (���磬41 �� 1)
    int size = markedData.size();
    for (int i = markedData.size() - marker_len; i < size; ++i) {
        markedData.push_back(1); // ����ʹ�ø����ӵı������
    }
    return markedData;
}

// ʹ�� Schifra ��ʵ�� RS ����
std::string WatermarkEncoder::performRSEncoding(const std::string& data) {
    // Schifraģ���������Ϊ�����ڳ���
    constexpr std::size_t field_descriptor = 8;
    constexpr std::size_t generator_polynomial_index = 120;
    constexpr std::size_t generator_polynomial_root_count = 32;

    constexpr std::size_t code_length = 255;
    constexpr std::size_t fec_length = 32;
    constexpr std::size_t data_length = 223;

    // 1. ����٤������
    const schifra::galois::field field(
        field_descriptor,
        schifra::galois::primitive_polynomial_size06,
        schifra::galois::primitive_polynomial06
    );

    // 2. �������ɶ���ʽ
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

    // 3. ���������
    typedef schifra::reed_solomon::encoder<code_length, fec_length, data_length> encoder_t;
    encoder_t encoder(field, generator_polynomial);

    std::string message(data);
    message.resize(code_length, 0x00); // ��䵽code_length

    // 5. ����RS��
    schifra::reed_solomon::block<code_length, fec_length> block;

    // 6. ����
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
        // �����λ�����λ������ȡ
        for (int b = 7; b >= 0; --b) {
            bits.push_back((ch >> b) & 0x01);
        }
    }
    return bits;
}

// ����ˮӡ (��Ӧ Step 3)
std::vector<int> WatermarkEncoder::encodeWatermark(const std::string& originalWatermark) {
    if (originalWatermark.empty()) {
        throw std::invalid_argument("Original watermark text cannot be empty.");
    }

    // 1. (ռλ) RS ����
    std::string encodedWatermark = performRSEncoding(originalWatermark);

    std::vector<int> encodedBits = stringToBits(encodedWatermark);

    // 2. ��ӱ����Ϣ
    std::vector<int> finalBits = addMarkerBits(encodedBits);

    /*for (auto& i : finalBits) {
        std::cout << i;
    }
    std::cout << std::endl;*/

    // ������ճ����Ƿ���� (���磬����Ϊ 0)
    if (finalBits.empty()) {
         throw std::runtime_error("Watermark encoding resulted in an empty bit sequence.");
    }

    return finalBits;
}
