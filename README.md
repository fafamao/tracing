# Ray Tracing Project 🚀

A high-performance ray tracer built with C++ and CUDA to generate photorealistic images. This project simulates the physics of light to render scenes with complex interactions, including reflections, refractions, and soft shadows.

![Rendered Image Placeholder]([https://via.placeholder.com/800x400.png?text=Your+Amazing+Render+Goes+Here](https://github.com/fafamao/tracing/blob/main/figure/image.png))

***

## 🌟 Features

* **CUDA-Accelerated:** Leverages the power of NVIDIA GPUs for massively parallel ray tracing computations.
* **Physically-Based Rendering (PBR):** Simulates real-world materials and lighting.
* **Material System:** Supports various materials like diffuse (Lambertian), metal (reflective), and dielectric (refractive).
* **Camera Model:** Implements a configurable camera with depth of field (defocus blur).
* **Anti-Aliasing:** Uses supersampling to smooth out jagged edges.
* **Bounding Volume Hierarchy (BVH):** Accelerates ray-object intersection tests for complex scenes.

***

## Benchmark data
TBD

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

## 🏃‍♀️ Usage

