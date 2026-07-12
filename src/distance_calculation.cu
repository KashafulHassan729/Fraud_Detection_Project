#include "distance_calculation.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <future>
#include <omp.h>

#define TILE_ROWS 256

// ================= Module 2a: Naive baseline kernel =================

__global__ void distance_kernel_naive(
    const float* query,
    const float* historical,
    float* distances,
    int num_rows,
    int num_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    float sum = 0.0f;
    for (int i = 0; i < num_features; i++) {
        float diff = query[i] - historical[(size_t)idx * num_features + i];
        sum += diff * diff;
    }
    distances[idx] = sqrtf(sum);
}

// ================= Module 2b: Tiled shared-memory kernel =================

__global__ void distance_kernel_tiled(
    const float* __restrict__ query,
    const float* __restrict__ historical,
    float* __restrict__ distances,
    int num_rows,
    int num_features
) {
    extern __shared__ float smem[];
    float* s_query = smem;                    // size: num_features
    float* s_tile  = smem + num_features;      // size: TILE_ROWS * num_features

    // Load query into shared memory
    for (int i = threadIdx.x; i < num_features; i += blockDim.x) {
        s_query[i] = query[i];
    }

    int row_block_start = blockIdx.x * TILE_ROWS;
    int tile_elems = TILE_ROWS * num_features;


    for (int e = threadIdx.x; e < tile_elems; e += blockDim.x) {
        int global_row = row_block_start + e / num_features;
        if (global_row < num_rows) {
            s_tile[e] = historical[(size_t)row_block_start * num_features + e];
        }
    }

    __syncthreads();

    int local_row  = threadIdx.x;
    int global_row = row_block_start + local_row;

    if (local_row < TILE_ROWS && global_row < num_rows) {
        float sum = 0.0f;
        for (int f = 0; f < num_features; f++) {
            float diff = s_query[f] - s_tile[local_row * num_features + f];
            sum += diff * diff;
        }
        distances[global_row] = sqrtf(sum);
    }
}

// ================= Host wrapper: naive =================
void compute_distances_cuda(
    const float* h_query,
    const float* h_historical,
    float* h_distances_out,
    int num_rows,
    int num_features
) {
    float *d_query, *d_historical, *d_distances;
    cudaMalloc(&d_query, num_features * sizeof(float));
    cudaMalloc(&d_historical, (size_t)num_rows * num_features * sizeof(float));
    cudaMalloc(&d_distances, num_rows * sizeof(float));

    cudaMemcpy(d_query, h_query, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_historical, h_historical, (size_t)num_rows * num_features * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    distance_kernel_naive<<<blocks, threads>>>(d_query, d_historical, d_distances, num_rows, num_features);
    cudaDeviceSynchronize();

    cudaMemcpy(h_distances_out, d_distances, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_historical);
    cudaFree(d_distances);
}

// ================= Host wrapper: tiled =================
void compute_distances_tiled(
    const float* h_query,
    const float* h_historical,
    float* h_distances_out,
    int num_rows,
    int num_features
) {
    float *d_query, *d_historical, *d_distances;
    cudaMalloc(&d_query, num_features * sizeof(float));
    cudaMalloc(&d_historical, (size_t)num_rows * num_features * sizeof(float));
    cudaMalloc(&d_distances, num_rows * sizeof(float));

    cudaMemcpy(d_query, h_query, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_historical, h_historical, (size_t)num_rows * num_features * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (num_rows + TILE_ROWS - 1) / TILE_ROWS;
    size_t shmem_bytes = (num_features + (size_t)TILE_ROWS * num_features) * sizeof(float);

    distance_kernel_tiled<<<blocks, TILE_ROWS, shmem_bytes>>>(d_query, d_historical, d_distances, num_rows, num_features);
    cudaDeviceSynchronize();

    cudaMemcpy(h_distances_out, d_distances, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_historical);
    cudaFree(d_distances);
}

// ================= Benchmark: naive vs tiled kernel time =================
void benchmark_naive_vs_tiled(
    const float* h_query,
    const float* h_historical,
    int num_rows,
    int num_features,
    float& naive_ms_out,
    float& tiled_ms_out
) {
    float *d_query, *d_historical, *d_distances;
    cudaMalloc(&d_query, num_features * sizeof(float));
    cudaMalloc(&d_historical, (size_t)num_rows * num_features * sizeof(float));
    cudaMalloc(&d_distances, num_rows * sizeof(float));
    cudaMemcpy(d_query, h_query, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_historical, h_historical, (size_t)num_rows * num_features * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Naive ---
    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    cudaEventRecord(start);
    distance_kernel_naive<<<blocks, threads>>>(d_query, d_historical, d_distances, num_rows, num_features);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naive_ms_out, start, stop);

    // --- Tiled ---
    int tiled_blocks = (num_rows + TILE_ROWS - 1) / TILE_ROWS;
    size_t shmem_bytes = (num_features + (size_t)TILE_ROWS * num_features) * sizeof(float);
    cudaEventRecord(start);
    distance_kernel_tiled<<<tiled_blocks, TILE_ROWS, shmem_bytes>>>(d_query, d_historical, d_distances, num_rows, num_features);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiled_ms_out, start, stop);

    printf("[benchmark] Naive kernel:  %.4f ms\n", naive_ms_out);
    printf("[benchmark] Tiled kernel:  %.4f ms\n", tiled_ms_out);
    printf("[benchmark] Speedup (naive/tiled): %.2fx\n", naive_ms_out / tiled_ms_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_query);
    cudaFree(d_historical);
    cudaFree(d_distances);
}

// ================= Benchmark: pinned vs pageable H2D transfer =================
void benchmark_pinned_vs_pageable(
    const float* h_historical_source,
    int num_rows,
    int num_features,
    float& pageable_ms_out,
    float& pinned_ms_out
) {
    size_t bytes = (size_t)num_rows * num_features * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Pageable (regular malloc/new, OS can swap it) ---
    float* h_pageable = (float*)malloc(bytes);
    memcpy(h_pageable, h_historical_source, bytes);
    float* d_buf_a;
    cudaMalloc(&d_buf_a, bytes);

    cudaEventRecord(start);
    cudaMemcpy(d_buf_a, h_pageable, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&pageable_ms_out, start, stop);

    // --- Pinned (page-locked, DMA-able) ---
    float* h_pinned;
    cudaHostAlloc((void**)&h_pinned, bytes, cudaHostAllocDefault);
    memcpy(h_pinned, h_historical_source, bytes);
    float* d_buf_b;
    cudaMalloc(&d_buf_b, bytes);

    cudaEventRecord(start);
    cudaMemcpy(d_buf_b, h_pinned, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&pinned_ms_out, start, stop);

    printf("[benchmark] Pageable H2D copy: %.4f ms (%.2f MB)\n", pageable_ms_out, bytes / 1e6);
    printf("[benchmark] Pinned   H2D copy: %.4f ms (%.2f MB)\n", pinned_ms_out, bytes / 1e6);
    printf("[benchmark] Pinned speedup: %.2fx\n", pageable_ms_out / pinned_ms_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_pageable);
    cudaFreeHost(h_pinned);
    cudaFree(d_buf_a);
    cudaFree(d_buf_b);
}

// ================= Module 4: Hybrid batched + streamed pipeline =================

static void prepare_batch_cpu(
    const float* h_source,
    float* h_pinned_dst,
    int row_start,
    int rows_this_batch,
    int num_features
) {
   
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_this_batch; r++) {
        for (int f = 0; f < num_features; f++) {
            size_t idx = (size_t)(row_start + r) * num_features + f;
            float v = h_source[idx];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            h_pinned_dst[(size_t)r * num_features + f] = v;
        }
    }
}

void compute_distances_hybrid_streamed(
    const float* h_query,
    const float* h_source,
    float* h_distances_out,
    int num_rows,
    int num_features,
    int batch_size,
    float& total_kernel_ms_out,
    float& total_transfer_ms_out,
    float& total_cpu_prep_ms_out
) {
    const int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) cudaStreamCreate(&streams[s]);

    float* d_query;
    cudaMalloc(&d_query, num_features * sizeof(float));
    cudaMemcpy(d_query, h_query, num_features * sizeof(float), cudaMemcpyHostToDevice);

    // Double-buffered PINNED staging (CPU writes here, GPU reads via async copy)
    float* h_pinned_buf[NUM_STREAMS];
    float* d_historical_buf[NUM_STREAMS];
    float* d_distances_buf[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaHostAlloc((void**)&h_pinned_buf[s], (size_t)batch_size * num_features * sizeof(float), cudaHostAllocDefault);
        cudaMalloc(&d_historical_buf[s], (size_t)batch_size * num_features * sizeof(float));
        cudaMalloc(&d_distances_buf[s], batch_size * sizeof(float));
    }

    int num_batches = (num_rows + batch_size - 1) / batch_size;

    cudaEvent_t xfer_start[NUM_STREAMS], xfer_stop[NUM_STREAMS];
    cudaEvent_t kern_start[NUM_STREAMS], kern_stop[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaEventCreate(&xfer_start[s]); cudaEventCreate(&xfer_stop[s]);
        cudaEventCreate(&kern_start[s]); cudaEventCreate(&kern_stop[s]);
    }

    total_kernel_ms_out = 0.0f;
    total_transfer_ms_out = 0.0f;
    total_cpu_prep_ms_out = 0.0f;

    auto batch_rows = [&](int b) {
        int row_start = b * batch_size;
        return std::min(batch_size, num_rows - row_start);
    };

    // Prepare batch 0 synchronously (nothing to overlap with yet)
    double t0 = omp_get_wtime();
    prepare_batch_cpu(h_source, h_pinned_buf[0], 0, batch_rows(0), num_features);
    total_cpu_prep_ms_out += (omp_get_wtime() - t0) * 1000.0;

    for (int b = 0; b < num_batches; b++) {
        int slot = b % NUM_STREAMS;
        int next_slot = (b + 1) % NUM_STREAMS;
        cudaStream_t s = streams[slot];

        int row_start = b * batch_size;
        int rows_this_batch = batch_rows(b);
        size_t bytes_this_batch = (size_t)rows_this_batch * num_features * sizeof(float);

        // Launch CPU prep for batch N+1 on a background thread FIRST,
        
        std::future<double> cpu_future;
        bool has_next = (b + 1 < num_batches);
        if (has_next) {
            int next_row_start = (b + 1) * batch_size;
            int next_rows = batch_rows(b + 1);
            cpu_future = std::async(std::launch::async, [&, next_row_start, next_rows, next_slot]() -> double {
                double t_start = omp_get_wtime();
                prepare_batch_cpu(h_source, h_pinned_buf[next_slot], next_row_start, next_rows, num_features);
                return (omp_get_wtime() - t_start) * 1000.0;
            });
        }

        // GPU side for CURRENT batch
        cudaEventRecord(xfer_start[slot], s);
        cudaMemcpyAsync(
            d_historical_buf[slot],
            h_pinned_buf[slot],
            bytes_this_batch,
            cudaMemcpyHostToDevice,
            s
        );
        cudaEventRecord(xfer_stop[slot], s);

        int blocks = (rows_this_batch + TILE_ROWS - 1) / TILE_ROWS;
        size_t shmem_bytes = (num_features + (size_t)TILE_ROWS * num_features) * sizeof(float);

        cudaEventRecord(kern_start[slot], s);
        distance_kernel_tiled<<<blocks, TILE_ROWS, shmem_bytes, s>>>(
            d_query, d_historical_buf[slot], d_distances_buf[slot], rows_this_batch, num_features
        );
        cudaEventRecord(kern_stop[slot], s);

        cudaMemcpyAsync(
            h_distances_out + row_start,
            d_distances_buf[slot],
            rows_this_batch * sizeof(float),
            cudaMemcpyDeviceToHost,
            s
        );

        
        if (has_next) {
            total_cpu_prep_ms_out += cpu_future.get();
        }
    }

    for (int s = 0; s < NUM_STREAMS; s++) cudaStreamSynchronize(streams[s]);

    for (int s = 0; s < NUM_STREAMS; s++) {
        float t_ms;
        cudaEventElapsedTime(&t_ms, xfer_start[s], xfer_stop[s]);
        total_transfer_ms_out += t_ms;
        cudaEventElapsedTime(&t_ms, kern_start[s], kern_stop[s]);
        total_kernel_ms_out += t_ms;
    }

    printf("[hybrid] Batches: %d | Batch size: %d\n", num_batches, batch_size);
    printf("[hybrid] Total CPU prep time (overlapped, logged): %.4f ms\n", total_cpu_prep_ms_out);
    printf("[hybrid] Total async transfer time (logged): %.4f ms\n", total_transfer_ms_out);
    printf("[hybrid] Total kernel time (logged): %.4f ms\n", total_kernel_ms_out);

    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaEventDestroy(xfer_start[s]); cudaEventDestroy(xfer_stop[s]);
        cudaEventDestroy(kern_start[s]); cudaEventDestroy(kern_stop[s]);
        cudaFreeHost(h_pinned_buf[s]);
        cudaFree(d_historical_buf[s]);
        cudaFree(d_distances_buf[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_query);
}
