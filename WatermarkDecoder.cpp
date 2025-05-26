#include "WatermarkDecoder.h"
#include <stdexcept>
#include "schifra_galois_field.hpp"
#include "schifra_galois_field_polynomial.hpp"
#include "schifra_sequential_root_generator_polynomial_creator.hpp"
#include "schifra_reed_solomon_encoder.hpp"
#include "schifra_reed_solomon_decoder.hpp"
#include "schifra_reed_solomon_block.hpp"
#include "schifra_error_processes.hpp"
#include <iostream>
#include <numeric>
#include <sstream>

WatermarkDecoder::WatermarkDecoder(int rsN, int rsK, int markerLength, double markerThreshold)
    : rs_n(rsN), rs_k(rsK), marker_len(markerLength), marker_correct_threshold(markerThreshold) {
    if (marker_len <= 0) {
        throw std::invalid_argument("Marker length must be positive.");
    }
    if (markerThreshold < 0.0 || markerThreshold > 1.0) {
        throw std::invalid_argument("Marker threshold must be between 0.0 and 1.0.");
    }
    // ��ʵ��Ӧ���У�������ʼ�� RS ��������
}

// �����λ��ȷ�� (������λ��ȫ 1)
bool WatermarkDecoder::checkMarkerBits(const std::vector<int>& bits) {
    if (bits.size() < marker_len) {
        std::cerr << "Error: Extracted bits are shorter than marker length." << std::endl;
        return false; // �����Լ����
    }

    int correctCount = 0;
    for (int i = bits.size() - marker_len; i < bits.size(); ++i) {
        if (bits[i] != 1) { // ����Ƕ��ʱ���λ�� 1
            correctCount++;
        }
    }

    double correctRate = static_cast<double>(correctCount) / marker_len;
    std::cout << "Marker bit correct rate: " << correctRate * 100 << "%" << std::endl;
    return correctRate <= marker_correct_threshold;
}

// ִ�� RS ���� (ռλ��)
std::string WatermarkDecoder::performRSDecoding(const std::string& data) {
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
    typedef schifra::reed_solomon::decoder<code_length, fec_length, data_length> decoder_t;
    const decoder_t decoder(field, generator_polynomial_index);


    std::string temp = data.substr(0, 8).append(std::string(215, '\0'));
    // 5. ����RS��
    schifra::reed_solomon::block<code_length, fec_length> block(temp, data.substr(8, 40));

    // 6. ����
    if (!decoder.decode(block)){
        std::cout << "Error - Critical decoding failure!" << std::endl;
        return data;
    }

    std::string decodeword;
    decodeword.resize(data_length);

    block.data_to_string(decodeword);

    return decodeword;
}

// ��������λתΪ�ַ��� (ÿ 8 bits תһ�� char)
std::string WatermarkDecoder::bitsToString(const std::vector<int>& bits) {
    std::string text = "";
    int n = bits.size();
    if (n % 8 != 0) {
        std::cerr << "Warning: Decoded bit count (" << n << ") is not a multiple of 8. String conversion might be incorrect." << std::endl;
    }

    for (int i = 0; i < n / 8; ++i) {
        char c = 0;
        for (int j = 0; j < 8; ++j) {
            if (bits[i * 8 + j] == 1) {
                c |= (1 << (7 - j));
            }
        }
        text += c;
    }
    return text;
}


// ������ȡ���ı����� (��Ӧ Step 4)
std::string WatermarkDecoder::decodeWatermark(std::vector<int>& extractedBits) {
    if (extractedBits.empty()) {
        throw std::runtime_error("No bits extracted to decode.");
    }

    /*for (auto& i : extractedBits) {
        std::cout << i;
    }
    std::cout << std::endl;*/

    // 1. �����λ
    if (!checkMarkerBits(extractedBits)) {
        throw std::runtime_error("Marker check failed. Cannot decode watermark reliably.");
    }

    extractedBits.resize(extractedBits.size() - 41);
    // 2. ������������λת��Ϊ�ַ���
    std::string originalWatermark = bitsToString(extractedBits);

    // 3. (ռλ) RS ���� (ȥ�����λ�������)
    std::string decodedData = performRSDecoding(originalWatermark);
    if (decodedData.empty()) {
         // ���� RS ����ʧ�ܻ�û������λ
         std::cerr << "Warning: RS decoding resulted in empty data bits." << std::endl;
         return "";
    }

    return decodedData;
}
