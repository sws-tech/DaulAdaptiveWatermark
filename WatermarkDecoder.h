#ifndef WATERMARK_DECODER_H
#define WATERMARK_DECODER_H

#include <vector>
#include <string>

class WatermarkDecoder {
public:
    // 构造函数，可以指定 RS 编码参数和标记信息长度
    WatermarkDecoder(int rsN = 255, int rsK = 223, int markerLength = 41, double markerThreshold = 0.2);

    // 解码提取出的比特流 (对应 Step 4)
    // 返回解码后的原始水印文本
    std::string decodeWatermark(std::vector<int>& extractedBits);

private:
    int rs_n; // RS 码总长度
    int rs_k; // RS 码信息位长度
    int marker_len; // 标记信息长度
    double marker_correct_threshold; // 标记位错误率阈值 (例如 0.2)

    // 内部函数：检查标记位正确率
    bool checkMarkerBits(const std::vector<int>& bits);

    // 内部函数：执行 RS 解码 (此处为简化示意，实际需要 RS 库)
    std::string performRSDecoding(const std::string& data);

    // 内部函数：将二进制位转为字符串
    std::string bitsToString(const std::vector<int>& bits);
};

#endif // WATERMARK_DECODER_H
