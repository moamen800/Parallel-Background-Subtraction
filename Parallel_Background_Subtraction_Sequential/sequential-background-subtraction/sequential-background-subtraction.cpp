#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <chrono> // ⏱️ For measuring execution time


// === Struct to hold video metadata ===
struct VideoMeta {
    int totalFrames, width, height;
    double fps;
};

// === Step 1: Read metadata from video ===
VideoMeta readVideoMeta(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened())
        throw std::runtime_error("Cannot open video: " + path);

    VideoMeta meta;
    meta.totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    meta.fps = cap.get(cv::CAP_PROP_FPS);
    meta.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    meta.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    return meta;
}

// === Step 2: Accumulate grayscale sum of all video frames ===
void computeSum(const std::string& path, cv::Mat& sum) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened())
        throw std::runtime_error("Cannot open video: " + path);

    cv::Mat frame, gray;
    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat grayDouble;
        gray.convertTo(grayDouble, CV_64F);
        sum += grayDouble;
    }
}

// === Step 3: Generate background image and binary foreground video ===
void generateOutput(const std::string& path, const cv::Mat& sum,
    int totalFrames, double fps, double threshold) {
    int h = sum.rows, w = sum.cols;

    // Compute average (background) image
    cv::Mat meanBg;
    sum.convertTo(meanBg, CV_8U, 1.0 / totalFrames);
    if (!cv::imwrite("output/background.png", meanBg))
        throw std::runtime_error("Cannot write background.png");

    // Setup video writer for grayscale binary foreground
    cv::VideoWriter writer("output/foreground.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps, cv::Size(w, h), false);
    if (!writer.isOpened())
        throw std::runtime_error("Failed to open foreground.mp4 for writing");

    // Replay original video and compute thresholded foreground
    cv::VideoCapture cap(path);
    if (!cap.isOpened())
        throw std::runtime_error("Cannot reopen video for foreground output");

    cv::Mat frame, gray, diff, mask;
    for (int i = 0; i < totalFrames; ++i) {
        if (!cap.read(frame)) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::absdiff(gray, meanBg, diff);
        cv::threshold(diff, mask, threshold, 255, cv::THRESH_BINARY);
        writer.write(mask);
    }
}

//int main() {
//    // === Step 4: Absolute path to video and output files ===
//    std::string inputPath = "C:/Users/sarah/Documents/SENIOR-2/SEMESTER-10/HPC/sequential-background-subtraction/input/dataset_video.mp4";
//    std::string outputBg = "output/background.png";
//    std::string outputFg = "output/foreground.mp4";
//    double threshold = 30.0; // ← Adjust this as needed
//
//    try {
//        // Step 5.1: Get video metadata
//        VideoMeta meta = readVideoMeta(inputPath);
//        std::cout << "Video loaded → " << meta.totalFrames << " frames @ " << meta.fps << " FPS\n";
//
//        // Step 5.2: Create empty grayscale accumulator
//        cv::Mat sum = cv::Mat::zeros(meta.height, meta.width, CV_64F);
//        computeSum(inputPath, sum);
//
//        // Step 5.3: Generate background + foreground output
//        generateOutput(inputPath, sum, meta.totalFrames, meta.fps, threshold);
//
//        std::cout << "✅ Done! Output files:\n";
//        std::cout << " → " << outputBg << "\n";
//        std::cout << " → " << outputFg << "\n";
//    }
//    catch (const std::exception& ex) {
//        std::cerr << "❌ Error: " << ex.what() << "\n";
//        return 1;
//    }

#include <filesystem> // Add this at the top with your includes

int main() {
    // === Step 4: Absolute path to video and output files ===
    //std::string inputPath = "C:/Users/sarah/Documents/SENIOR-2/SEMESTER-10/HPC/sequential-background-subtraction/input/dataset_video.mp4";
    namespace fs = std::filesystem;
    std::string inputDir = "C:/Users/sarah/Documents/SENIOR-2/SEMESTER-10/HPC/sequential-background-subtraction/input/";
    std::string inputPath;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".mp4") {
            inputPath = entry.path().string();
            break;
        }
    }

    if (inputPath.empty()) {
        throw std::runtime_error("No .mp4 file found in input folder!");
    }

    std::string outputBg = "output/background.png";
    std::string outputFg = "output/foreground.mp4";
    double threshold = 30.0; // ← Adjust this as needed

    try {
        // ✅ Ensure the "output" directory exists
        std::filesystem::create_directories("output");

        // ⏱️ Start measuring time
        auto start = std::chrono::high_resolution_clock::now();

        // Step 5.1: Get video metadata
        VideoMeta meta = readVideoMeta(inputPath);
        std::cout << "Video loaded → " << meta.totalFrames << " frames @ " << meta.fps << " FPS\n";

        // Step 5.2: Create empty grayscale accumulator
        cv::Mat sum = cv::Mat::zeros(meta.height, meta.width, CV_64F);
        computeSum(inputPath, sum);

        // Step 5.3: Generate background + foreground output
        generateOutput(inputPath, sum, meta.totalFrames, meta.fps, threshold);

        // ⏱️ Stop measuring time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;


        std::cout << "✅ Done! Output files:\n";
        std::cout << " → " << outputBg << "\n";
        std::cout << " → " << outputFg << "\n";
        std::cout << "🕒 Total processing time: " << duration.count() << " seconds\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "❌ Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}


