#include <iostream>                          // for std::cout, std::cerr
#include <string>                            // for std::string
#include <mpi.h>                             // MPI functions
#include "mpi_processing/MPIProcessor.h"     // our MPI orchestration module

// Default threshold for foreground detection
static constexpr double DEFAULT_THRESHOLD = 30.0;

int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Determine this process's rank (ID) and the total number of processes
    int rank{}, size{};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Record the start time of the MPI run (high-resolution wall-clock)
    double t_start = MPI_Wtime();

    // Expect exactly 3 arguments: input video, background output, foreground output
    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -n <P> ./bg_subtract"
                      << " <input.mp4> <out_bg.png> <out_fg.mp4> "
                      << "(uses default threshold=" << DEFAULT_THRESHOLD << ")\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Parse command-line arguments (skip threshold)
    std::string inputVid = argv[1];
    std::string outBg    = argv[2];
    std::string outFg    = argv[3];
    double threshold     = DEFAULT_THRESHOLD;

    // Execute MPI-based background subtraction
    int rc = MPIProcessor::run(
        threshold,
        inputVid,
        outBg,
        outFg,
        rank,
        size
    );

    // Measure end time
    double t_end = MPI_Wtime();
    MPI_Finalize();

    if (rank == 0) {
        std::cout << "Total MPI run time: " << (t_end - t_start) << " seconds\n";
    }
    return rc;
}
