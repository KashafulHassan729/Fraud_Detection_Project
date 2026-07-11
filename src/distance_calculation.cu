#include "distance_calculation.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void distance_kernel(
    const float* query,
    const float* historical,
    float* distances,
    int num_rows,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    // TODO: replace with tiled shared-memory version (Module 2 requirement)
    float sum = 0.0f;
    for (int i = 0; i < num_features; i++) {
        float diff = query[i] - historical[idx * num_features + i];
        sum += diff * diff;
    }
    distances[idx] = sqrtf(sum);
}

void compute_distances_cuda(
    const float* h_query,
    const float* h_historical,
    float* h_distances_out,
    int num_rows,
    int num_features
) {
    std::cout << "[distance_calculation] STUB: launching naive kernel" << std::endl;

    float *d_query, *d_historical, *d_distances;
    cudaMalloc(&d_query, num_features * sizeof(float));
    cudaMalloc(&d_historical, (size_t)num_rows * num_features * sizeof(float));
    cudaMalloc(&d_distances, num_rows * sizeof(float));

    cudaMemcpy(d_query, h_query, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_historical, h_historical, (size_t)num_rows * num_features * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    distance_kernel<<<blocks, threads>>>(d_query, d_historical, d_distances, num_rows, num_features);
    cudaDeviceSynchronize();

    cudaMemcpy(h_distances_out, d_distances, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_historical);
    cudaFree(d_distances);
}