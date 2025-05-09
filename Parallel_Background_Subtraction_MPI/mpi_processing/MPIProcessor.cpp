// mpi_processing/MPIProcessor.cpp
#include "MPIProcessor.h"
#include <iostream>
#include <exception>

namespace MPIProcessor {

/**
 * Entry point for the MPI-based background subtraction processor.
 * @param thresh   Threshold value for foreground detection.
 * @param inVid    Path to the input video file.
 * @param outBg    Filename for the generated background video.
 * @param outFg    Filename for the generated foreground video.
 * @param rank     MPI rank of this process.
 * @param size     Total number of MPI processes.
 * @return         Status code (0 = success, non-zero = error).
 */
int run(double thresh,
        const std::string& inVid,
        const std::string& outBg,
        const std::string& outFg,
        int rank,
        int size)
{
    VideoMeta meta;

    // Only rank 0 reads the video metadata to avoid redundant I/O
    if (rank == 0) {
        try {
            meta = readVideoMeta(inVid);  // Extract totalFrames, fps, width, height
        } catch (std::exception &e) {
            std::cerr << "Error reading video: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);    // Abort all processes on failure
        }
    }

    // Broadcast metadata from rank 0 to all other ranks
    MPI_Bcast(&meta.totalFrames, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&meta.fps,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meta.width,       1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&meta.height,      1, MPI_INT,    0, MPI_COMM_WORLD);

    // Initialize a local accumulator for this rank's frame subset
    cv::Mat localSum = cv::Mat::zeros(meta.height, meta.width, CV_64F);
    try {
        // Each rank processes its share of frames to compute a per-pixel sum
        computeLocalSum(inVid, rank, size, localSum);
    } catch (std::exception &e) {
        if (rank == 0) std::cerr << "Error in computeLocalSum: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Prepare a matrix on rank 0 to collect the global sum
    cv::Mat globalSum = cv::Mat::zeros(meta.height, meta.width, CV_64F);
    // Sum up all localSum matrices into globalSum on rank 0
    MPI_Reduce(localSum.ptr<double>(),         // send buffer
               globalSum.ptr<double>(),        // receive buffer (only valid on root)
               meta.height * meta.width,       // element count
               MPI_DOUBLE,                     // data type
               MPI_SUM,                        // operation
               0,                              // root rank
               MPI_COMM_WORLD);

    // Rank 0 generates the final background and foreground outputs
    if (rank == 0) {
        try {
            generateOutputs(
                inVid,                // original video
                globalSum,            // summed pixel values
                meta.totalFrames,     // total number of frames
                meta.fps,             // frames per second
                thresh,               // threshold for foreground
                "output/" + outBg,   // output path for background
                "output/" + outFg    // output path for foreground
            );
            std::cout << "Done → output/" << outBg << ", output/" << outFg << "\n";
        } catch (std::exception &e) {
            std::cerr << "Error generating outputs: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;  // Success
}

} // namespace MPIProcessor
