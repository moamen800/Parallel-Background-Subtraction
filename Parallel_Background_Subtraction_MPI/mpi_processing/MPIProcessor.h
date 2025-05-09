// mpi_processing/MPIProcessor.h

#pragma once

#include <string>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include "../utils/VideoProcessor.h"

namespace MPIProcessor {

    /**
     * Runs the entire background subtraction workflow under MPI.
     *
     * @param thresh   threshold for FG detection
     * @param inVid    path to input video
     * @param outBg    filename for background image (written to output/)
     * @param outFg    filename for foreground video (written to output/)
     * @param rank     MPI rank (0..P-1)
     * @param size     MPI size (P)
     * @returns        0 on success, non-zero on error
     */
    int run(double thresh,
            const std::string& inVid,
            const std::string& outBg,
            const std::string& outFg,
            int rank,
            int size);

} // namespace MPIProcessor
