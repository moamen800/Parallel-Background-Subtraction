#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <omp.h>

namespace fs = std::filesystem;

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // === Dynamically detect any .mp4 file inside the input directory ===
    std::string video_path = "C:/Users/sarah/Documents/SENIOR-2/SEMESTER-10/HPC/openmp-background-subtraction/input/";
    std::string inputPath;
    for (const auto& entry : fs::directory_iterator(video_path)) {
        if (entry.path().extension() == ".mp4") {
            inputPath = entry.path().string();
            break;
        }
    }

    if (inputPath.empty()) {
        throw std::runtime_error("No .mp4 file found in input folder!");
    }

    std::string output_folder = "output";
    fs::create_directories(output_folder);

    // === OpenCV video capture ===
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "❌ Cannot open input video!\n";
        return -1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads.\n";
    std::cout << "Total video frames: " << total_frames << ", FPS: " << fps << "\n";

    // === STEP 1: Background Accumulation (Sequential) ===
    // This part is still sequential due to OpenCV's VideoCapture not being thread-safe.
    // We read frames, convert to grayscale, and sum them to compute the mean background.
    cv::Mat background_sum = cv::Mat::zeros(height, width, CV_64F);
    std::vector<cv::Mat> gray_frames;

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gray_frames.push_back(gray.clone());

        cv::Mat gray64f;
        gray.convertTo(gray64f, CV_64F);
        background_sum += gray64f;
    }
    cap.release();

    int actual_frames = gray_frames.size();
    cv::Mat background;
    background_sum.convertTo(background, CV_8U, 1.0 / actual_frames);
    cv::imwrite(output_folder + "/estimated_background.jpg", background);
    std::cout << "✅ Saved background image\n";

    // === STEP 2: Foreground Mask Computation (Parallel) ===
    // We parallelize the loop using OpenMP. Each thread processes one frame:
    // - Compute the absolute difference between frame and background.
    // - Apply thresholding to generate the binary foreground mask.
    // NOTE: Video writing is kept in a critical section due to OpenCV's thread-unsafety.
    double adjusted_fps = 3.0;
    cv::VideoWriter writer(output_folder + "/foreground.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), adjusted_fps, cv::Size(width, height), false);

    if (!writer.isOpened()) {
        std::cerr << "❌ Failed to open output video!\n";
        return -1;
    }

    // === OpenMP parallel loop for foreground computation ===
#pragma omp parallel for
    for (int i = 0; i < actual_frames; ++i) {
        if (gray_frames[i].empty()) continue;

        cv::Mat frame_gray = gray_frames[i];
        cv::Mat diff, mask;

        cv::absdiff(frame_gray, background, diff);
        cv::threshold(diff, mask, 30, 255, cv::THRESH_BINARY);

        // Writing must be serialized due to shared resource (writer)
#pragma omp critical
        writer.write(mask);
    }

    writer.release();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "✅ Foreground video written to: " << output_folder << "/foreground.mp4\n";
    std::cout << "🕒 Total processing time: " << elapsed.count() << " seconds.\n";

    return 0;
}
