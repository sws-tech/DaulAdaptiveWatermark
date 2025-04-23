#ifndef WATERMARK_DECODER_H
#define WATERMARK_DECODER_H

#include <vector>
#include <string>

class WatermarkDecoder {
public:
    // ���캯��������ָ�� RS ��������ͱ����Ϣ����
    WatermarkDecoder(int rsN = 255, int rsK = 223, int markerLength = 41, double markerThreshold = 0.2);

    // ������ȡ���ı����� (��Ӧ Step 4)
    // ���ؽ�����ԭʼˮӡ�ı�
    std::string decodeWatermark(std::vector<int>& extractedBits);

private:
    int rs_n; // RS ���ܳ���
    int rs_k; // RS ����Ϣλ����
    int marker_len; // �����Ϣ����
    double marker_correct_threshold; // ���λ��������ֵ (���� 0.2)

    // �ڲ������������λ��ȷ��
    bool checkMarkerBits(const std::vector<int>& bits);

    // �ڲ�������ִ�� RS ���� (�˴�Ϊ��ʾ�⣬ʵ����Ҫ RS ��)
    std::string performRSDecoding(const std::string& data);

    // �ڲ���������������λתΪ�ַ���
    std::string bitsToString(const std::vector<int>& bits);
};

#endif // WATERMARK_DECODER_H
