#ifndef WATERMARK_ENCODER_H
#define WATERMARK_ENCODER_H

#include <vector>
#include <string>

class WatermarkEncoder {
public:
    // ���캯��������ָ�� RS ��������ͱ����Ϣ����
    WatermarkEncoder(int rsN = 255, int rsK = 223, int markerLength = 41); // ʾ�� RS(255, 223)

    // ����ˮӡ (��Ӧ Step 3)
    // ���ض�����ˮӡ���� (0 �� 1)
    std::vector<int> encodeWatermark(const std::string& originalWatermark);

private:
    int rs_n; // RS ���ܳ���
    int rs_k; // RS ����Ϣλ����
    int marker_len; // �����Ϣ����

    // �ڲ���������ӱ����Ϣ
    std::vector<int> addMarkerBits(const std::vector<int>& data);

    // �ڲ�������ִ�� RS ���� (�˴�Ϊ��ʾ�⣬ʵ����Ҫ RS ��)
    std::string performRSEncoding(const std::string& data);

    // �ڲ����������ַ���תΪ������λ
    std::vector<int> stringToBits(const std::string& text);
};

#endif // WATERMARK_ENCODER_H
