#include "VideoProcessor.h"
#include <stdexcept>

// Read video metadata (total frames, fps, width, height)
VideoMeta readVideoMeta(const std::string &path) {
    // Open the video file
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) 
        // Throw if the file cannot be opened
        throw std::runtime_error("Cannot open video: " + path);

    VideoMeta m;
    // Retrieve total number of frames
    m.totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    // Retrieve frames per second
    m.fps         = cap.get(cv::CAP_PROP_FPS);
    // Retrieve frame width
    m.width       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // Retrieve frame height
    m.height      = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Release the capture object
    cap.release();
    return m;
}

// Compute partial sum of grayscale frames for a given MPI rank
void computeLocalSum(const std::string &path, int rank, int size, cv::Mat &localSum) {
    // Open the video file
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) 
        // Throw if cannot open
        throw std::runtime_error("Cannot open video: " + path);

    cv::Mat frame, gray;
    // Get total frame count
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    // Iterate over frames assigned to this rank
    for (int i = rank; i < total; i += size) {
        // Seek to the i-th frame
        cap.set(cv::CAP_PROP_POS_FRAMES, i);
        cap.read(frame);
        if (frame.empty()) 
            // Error if frame is missing
            throw std::runtime_error("Empty frame #" + std::to_string(i));
        // Convert frame to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat tmp;
        // Convert to double for precise accumulation
        gray.convertTo(tmp, CV_64F);
        // Accumulate pixel values into localSum
        localSum += tmp;
    }
    // Release capture when done
    cap.release();
}

// Generate background image and foreground mask video from summed frames
void generateOutputs(const std::string &path,
                     const cv::Mat &globalSum,
                     int totalFrames,
                     double fps,
                     double threshold,
                     const std::string &bgOut,
                     const std::string &fgOut) {
    int h = globalSum.rows;
    int w = globalSum.cols;
    cv::Mat meanBg;
    // Compute mean background by scaling the sum
    globalSum.convertTo(meanBg, CV_8U, 1.0 / totalFrames);
    // Write the background image to file
    cv::imwrite(bgOut, meanBg);

    // Initialize video writer for foreground masks (single-channel)
    cv::VideoWriter writer(
        fgOut,
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(w, h),
        false // false = grayscale
    );
    if (!writer.isOpened())
        // Throw if writer cannot be created
        throw std::runtime_error("Cannot open writer: " + fgOut);

    // Re-open the video for reading frames
    cv::VideoCapture cap(path);
    cv::Mat frame, gray, diff, mask;
    // Process each frame to compute foreground mask
    for (int i = 0; i < totalFrames; ++i) {
        cap.set(cv::CAP_PROP_POS_FRAMES, i);
        cap.read(frame);
        if (frame.empty()) 
            // Stop if no more frames
            break;
        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // Compute absolute difference from background
        cv::absdiff(meanBg, gray, diff);
        // Threshold to create binary mask
        cv::threshold(diff, mask, threshold, 255, cv::THRESH_BINARY);
        // Write the mask frame to the output video
        writer.write(mask);
    }
    // Release resources
    cap.release();
    writer.release();
}
