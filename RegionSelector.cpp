#include "RegionSelector.h"
#include <algorithm> // for std::sort, std::stable_sort

RegionSelector::RegionSelector(RegionScorer scorer, int numRegionsToSelect, double windowScale, double stepScale)
    : regionScorer(scorer), targetRegionCount(numRegionsToSelect), windowSizeScale(windowScale), stepSizeScale(stepScale) {
    if (windowScale <= 0 || windowScale > 1 || stepScale <= 0 || stepScale > 1) {
        throw std::invalid_argument("RegionSelector: Window scale and step scale must be between 0 and 1.");
    }
     if (numRegionsToSelect <= 0) {
         throw std::invalid_argument("RegionSelector: Number of regions to select must be positive.");
     }
}

std::vector<Region> RegionSelector::selectEmbeddingRegions(const cv::Mat& originalImage, const cv::Mat& edgeImage) {
    if (originalImage.empty() || edgeImage.empty() || originalImage.size() != edgeImage.size()) {
        throw std::runtime_error("RegionSelector: Input images are invalid or mismatched.");
    }
    if (originalImage.channels() != 1 || edgeImage.channels() != 1) {
        throw std::runtime_error("RegionSelector: Images must be single-channel grayscale.");
    }

    int imgHeight = originalImage.rows;
    int imgWidth = originalImage.cols;
    cv::Point imageCenter(imgWidth / 2, imgHeight / 2);

    // 根据比例计算窗口大小 (确保至少为 1x1)
    int windowHeight = std::max(1, static_cast<int>(imgHeight * windowSizeScale));
    int windowWidth = std::max(1, static_cast<int>(imgWidth * windowSizeScale));

    // 根据比例计算步长 (确保至少为 1)
    int stepY = std::max(1, static_cast<int>(windowHeight * stepSizeScale));
    int stepX = std::max(1, static_cast<int>(windowWidth * stepSizeScale));

    std::vector<Region> candidateRegions;

    // 滑动窗口遍历
    for (int y = 0; y <= imgHeight - windowHeight; y += stepY) {
        for (int x = 0; x <= imgWidth - windowWidth; x += stepX) {
            Region currentRegion;
            currentRegion.bounds = cv::Rect(x, y, windowWidth, windowHeight);
            currentRegion.center = cv::Point(x + windowWidth / 2, y + windowHeight / 2);

            // 提取当前窗口对应的图像块
            cv::Mat originalPatch = originalImage(currentRegion.bounds);
            cv::Mat edgePatch = edgeImage(currentRegion.bounds);

            // 计算得分
            try {
                 regionScorer.calculateRegionScores(currentRegion, originalPatch, edgePatch, imageCenter);
                 candidateRegions.push_back(currentRegion);
            } catch (const std::exception& e) {
                // 可以记录日志或忽略计算失败的窗口
                 std::cerr << "Warning: Failed to score region at (" << x << "," << y << "): " << e.what() << std::endl;
            }
        }
    }

    // 按综合得分从高到低排序
    std::stable_sort(candidateRegions.begin(), candidateRegions.end(), [](const Region& a, const Region& b) {
        return a.score > b.score; // 降序
    });

    // 选择前 d 个非重叠区域
    std::vector<Region> selectedRegions;
    for (const auto& candidate : candidateRegions) {
        if (selectedRegions.size() >= targetRegionCount) {
            break; // 已选够数量
        }

        //bool overlaps = false;
        //for (auto& selected : selectedRegions) {
        //    if (candidate.overlaps(selected)) {
        //        if (selected.score < candidate.score) {
        //            selected = candidate;
        //        }
        //        overlaps = true;
        //        break;
        //    }
        //}
        //一个非常简单替换修改，但是感觉没什么用
        bool overlaps = false;
        for (const auto& selected : selectedRegions) {
            if (candidate.overlaps(selected)) {
                overlaps = true;
                break;
            }
        }

        if (!overlaps) {
            selectedRegions.push_back(candidate);
        }
    }

     if (selectedRegions.empty() && !candidateRegions.empty()) {
         // 如果严格非重叠选不到，至少返回得分最高的那个
         selectedRegions.push_back(candidateRegions[0]);
         std::cerr << "Warning: Could not find enough non-overlapping regions. Returning the highest scoring one(s)." << std::endl;
     } else if (selectedRegions.size() < targetRegionCount) {
         std::cerr << "Warning: Found only " << selectedRegions.size() << " non-overlapping regions (target was " << targetRegionCount << ")." << std::endl;
     }


    return selectedRegions;
}
