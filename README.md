# 🚀 Parallel-Background-Subtraction
It’s a technique for removing static background by subtracting a set of images to obtain a final image with the objects only without a background. This is a basic technique which will work only on tiny/small motion changes in the images that are given.

## 🟢 Project Description

This project implements **Background Subtraction** using three different approaches:
1. **Sequential (Single-threaded CPU approach)**
2. **OpenMP (Shared Memory Parallelism)**
3. **MPI (Message Passing Interface for Distributed Memory Parallelism)**

The purpose of the project is to compare the performance of these approaches for estimating background images and generating foreground masks from a video input.
