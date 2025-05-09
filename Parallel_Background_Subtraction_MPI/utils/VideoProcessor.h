#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>

struct VideoMeta {
    int totalFrames, width, height;
    double fps;
};

VideoMeta readVideoMeta(const std::string &path);
void computeLocalSum(const std::string &path,
                     int rank, int size,
                     cv::Mat &localSum);
void generateOutputs(const std::string &path,
                     const cv::Mat &globalSum,
                     int totalFrames, double fps,
                     double threshold,
                     const std::string &bgOut,
                     const std::string &fgOut);

#endif // VIDEO_PROCESSOR_H
