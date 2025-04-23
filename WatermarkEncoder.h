#ifndef WATERMARK_ENCODER_H
#define WATERMARK_ENCODER_H

#include <vector>
#include <string>

class WatermarkEncoder {
public:
    // 构造函数，可以指定 RS 编码参数和标记信息长度
    WatermarkEncoder(int rsN = 255, int rsK = 223, int markerLength = 41); // 示例 RS(255, 223)

    // 编码水印 (对应 Step 3)
    // 返回二进制水印序列 (0 或 1)
    std::vector<int> encodeWatermark(const std::string& originalWatermark);

private:
    int rs_n; // RS 码总长度
    int rs_k; // RS 码信息位长度
    int marker_len; // 标记信息长度

    // 内部函数：添加标记信息
    std::vector<int> addMarkerBits(const std::vector<int>& data);

    // 内部函数：执行 RS 编码 (此处为简化示意，实际需要 RS 库)
    std::string performRSEncoding(const std::string& data);

    // 内部函数：将字符串转为二进制位
    std::vector<int> stringToBits(const std::string& text);
};

#endif // WATERMARK_ENCODER_H
