# ------------------------------------------
# Compiler and flags
# ------------------------------------------
CXX      = mpicxx                                              # MPI C++ compiler
CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)     # C++17 + OpenCV include flags
LDFLAGS  = $(shell pkg-config --libs opencv4)                  # OpenCV library linker flags
INCLUDE  = -I./utils                                           # Additional include path for project headers

# ------------------------------------------
# Source files
# ------------------------------------------
SRC_ROOT  = main.cpp                                           # Entry point (root-level)
SRC_MPI   = mpi_processing/MPIProcessor.cpp                    # MPI orchestration module
SRC_UTILS = utils/VideoProcessor.cpp                           # Video processing utilities

# ------------------------------------------
# Object files (auto-derived from .cpp names)
# ------------------------------------------
OBJS = $(SRC_ROOT:.cpp=.o) $(SRC_MPI:.cpp=.o) $(SRC_UTILS:.cpp=.o)  # main.o, MPIProcessor.o, VideoProcessor.o

# ------------------------------------------
# Final executable name
# ------------------------------------------
TARGET = bg_subtract                                           # Name of the MPI binary

# ------------------------------------------
# Default target: build the executable
# ------------------------------------------
all: $(TARGET)

# ------------------------------------------
# Link step:  
#   - all object files  
# Produces the final binary with proper flags
# ------------------------------------------
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@ $(LDFLAGS)
	@echo "Built $(TARGET)"

# ------------------------------------------
# Compile rule for any .cpp → .o  
# $< = source, $@ = target object
# ------------------------------------------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@
	@echo "Compiled $< → $@"

# ------------------------------------------
# Clean up generated files
# ------------------------------------------
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Cleaned up objects and binary"




# # ------------------------------------------
# # Compiler and flags
# # ------------------------------------------
# CXX      = mpicxx                                              # MPI C++ compiler
# CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)     # C++17 + OpenCV include flags
# LDFLAGS  = $(shell pkg-config --libs opencv4)                  # OpenCV library linker flags
# INCLUDE  = -I./utils                                           # Additional include path for project headers

# # ------------------------------------------
# # Source files
# # ------------------------------------------
# SRC_ROOT  = main.cpp                                           # Entry point (root-level)
# SRC_MPI   = mpi_processing/MPIProcessor.cpp                    # MPI orchestration module
# SRC_UTILS = utils/VideoProcessor.cpp                           # Video processing utilities

# # ------------------------------------------
# # Object files (auto-derived from .cpp names)
# # ------------------------------------------
# OBJS = $(SRC_MPI:.cpp=.o) $(SRC_UTILS:.cpp=.o)                  # e.g. MPIProcessor.o, VideoProcessor.o

# # ------------------------------------------
# # Final executable name
# # ------------------------------------------
# TARGET = bg_subtract                                           # Name of the MPI binary

# # ------------------------------------------
# # Default target: build the executable
# # ------------------------------------------
# all: $(TARGET)

# # ------------------------------------------
# # Link step:  
# #   - main.cpp (no .o because it's small)  
# #   - all object files  
# # Produces the final binary with proper flags
# # ------------------------------------------
# $(TARGET): $(SRC_ROOT) $(OBJS)
# 	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@ $(LDFLAGS)
# 	@echo "Built $(TARGET)"

# # ------------------------------------------
# # Compile rule for any .cpp → .o  
# # $< = source, $@ = target object
# # ------------------------------------------
# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@
# 	@echo "Compiled $< → $@"

# # ------------------------------------------
# # Clean up generated files
# # ------------------------------------------
# clean:
# 	rm -f $(OBJS) $(TARGET)
# 	@echo "Cleaned up objects and binary"

