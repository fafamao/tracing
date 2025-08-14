# Ray Tracing Project 🚀

A high-performance ray tracer built with C++ and CUDA to generate photorealistic images. This project simulates the physics of light to render scenes with complex interactions, including reflections, refractions, and soft shadows.

![Rendered Image](./figure/image_pod.png)

***

## 🌟 Design

1.  **CPU Rendering with Multi-Threading:**

The initial implementation was based on **Peter Shirley**'s fantastic book, **Ray Tracing in One Weekend**. This served as the foundation for the core logic, including classes for vectors, rays, and materials. While functional, this single-threaded version was quite slow. Using **gprof** to identify bottlenecks and by implementing a **thread pool** to parallelize the rendering work across CPU cores, the rendering time for the final scene was reduced from minutes to under three seconds.

2.  **Initial GPU Port with CUDA:** 

To unlock massively parallel performance, the project was ported to **CUDA**. This first pass was a direct adaptation of the C++ object-oriented design. While it successfully moved the rendering workload to the GPU, it retained the original class structures, including **virtual functions** for material dispatching. This approach, while functional, is not optimal for the GPU architecture due to the overhead associated with virtual table lookups, which can hinder performance in parallel workloads.

3.  **High-Performance GPU Rendering with POD:**

The final and most significant optimization was a complete refactor of the data structures to a **Plain Old Data (POD)** format. All object-oriented patterns, especially virtual functions, were replaced with a more GPU-friendly approach using enums and function tables for dispatching. This transformation ensures that all data sent to the GPU is simple, contiguous, and free of hidden overhead, allowing the hardware to achieve maximum throughput. This extensive refactoring resulted in a substantial performance gain and represents a standard, high-performance pattern for modern CUDA applications.

***

## 🚀 Performance

Here is a comparison of the final render times for a 1280x720 image with 10 samples per pixel, running on an NVIDIA RTX 3060 and Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz.

| Implementation          | Time to Render | Notes                               |
| ----------------------- | :------------: | ----------------------------------- |
| CPU (Multi-threaded)    |   **11586 ms**   | Optimized with a thread pool.       |
| GPU (Initial CUDA Port) |   **1137.9 ms**       | A direct port of the C++ classes.   |
| GPU (POD Refactor)      |   **311.445496 ms** | Fully optimized with a POD design.  |
| GPU (POD Refactor + Texture Cache)      |   **341.315582 ms** | Texture cache is slowing it down.  |

The final POD-based GPU implementation demonstrates a significant speedup over both the multi-threaded CPU version and the initial, non-optimized CUDA port.

### Bottleneck

After profiling with ncu, the main concern is the low warp occupancy and high register usage which should be the direct cause of the previous issue.

| Item          | Data |
| ----------------------- | :------------: |
| Registers Per Thread    |   **92**   |
| Theoretical Occupancy |   **33.33%**       |
| Theoretical Active Warps per SM      |   **16** |
| Achieved Occupancy      |   **27.52** |
| Achieved Active Warps per SM      |   **13.21** |

RTX 3050 has below capability

* Max Threads per SM: 1536
* Max Thread Blocks per SM: 16
* Max Warps per SM: 48
* Max Register File Size: 65,536 registers
* Max Shared Memory: 100 KB

The main **render_kernel** has block dimension (16,16) and grid size(80,45), so
1. warps per block
   16 * 16 / 32 = 8 warps
2. maximum blocks per SM
   1536 / (16 * 16) = 6 blocks
3. warps per SM
   8 * 6 = 48 warps

RTX3050 with Compute Capability 8.6 would support 48 warps per SM so threading alone would achieve 100% occupancy. However, registers used per thread is **92** which further limits the occupancy.

1. registers per block
   92 * 16 * 16 = 23552 registers
2. blocks per SM
   65536 / 23552 = 2.78 blocks
3. new occupancy
   2 * 8 / 48 = 33.33%

## 🛠️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

Make sure you have the following installed:

* **NVIDIA GPU:** A CUDA-enabled NVIDIA GPU is required.
* **NVIDIA CUDA Toolkit:** [Download and install the CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
* **C++ Compiler:** A C++ compiler that is compatible with your CUDA version (e.g., GCC, Clang, or MSVC).
* **Xmake:** https://xmake.io/#/getting_started.

### Installation & Building

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/fafamao/tracing.git](https://www.google.com/search?q=https://github.com/fafamao/tracing.git)
    ```

2.  **Build the program:**
    ```sh
    xmake build tracing
    ```

3.  **Run the program:**
    ```sh
    xmake run tracing
    ```
***

