#include "distance_calculation.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#define TILE_ROWS 256

// ================= Module 2a: Naive baseline kernel =================
// 1 thread computes 1 distance. Each thread strides through global
// memory with stride = num_features between consecutive threads'
// accesses at a fixed feature index -> NOT coalesced. This is the
// "bad" version we compare the tiled kernel against.
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
// Blocking strategy:
//   - Each block owns a tile of TILE_ROWS historical rows (row blocking).
//   - The block cooperatively loads that whole tile (all num_features
//     columns for those rows) into shared memory in ONE pass, using a
//     flat index so consecutive threads read consecutive global memory
//     addresses -> fully coalesced load.
//   - After __syncthreads(), each thread computes its own row's distance
//     by reading from fast on-chip shared memory instead of global memory.
// The query vector is also cached in shared memory once per block.
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

    // Load query into shared memory (cheap, done by every block once)
    for (int i = threadIdx.x; i < num_features; i += blockDim.x) {
        s_query[i] = query[i];
    }

    int row_block_start = blockIdx.x * TILE_ROWS;
    int tile_elems = TILE_ROWS * num_features;

    // Coalesced cooperative load: historical is row-major, so
    // historical[row_block_start*num_features + e] is contiguous as e
    // increases. Consecutive threadIdx.x values -> consecutive addresses.
    for (int e = threadIdx.x; e < tile_elems; e += blockDim.x) {
        int global_row = row_block_start + e / num_features;
        if (global_row < num_rows) {
            s_tile[e] = historical[(size_t)row_block_start * num_features + e];
        }
    }

    __syncthreads(); // MUST finish loading before any thread reads s_tile

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
// Double-buffered, 2-stream design:
//   - Historical data is split into batches of `batch_size` rows.
//   - Stream A and Stream B alternate: while stream X runs the tiled
//     kernel on batch N, stream Y issues the async H2D copy for batch
//     N+1 in parallel (this is the actual "overlap compute with next
//     batch's transfer" requirement).
//   - h_historical_pinned MUST be pinned (cudaHostAlloc) or
//     cudaMemcpyAsync silently falls back to synchronous behavior.
void compute_distances_hybrid_streamed(
    const float* h_query,
    const float* h_historical_pinned,
    float* h_distances_out,
    int num_rows,
    int num_features,
    int batch_size,
    float& total_kernel_ms_out,
    float& total_transfer_ms_out
) {
    const int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) cudaStreamCreate(&streams[s]);

    float* d_query;
    cudaMalloc(&d_query, num_features * sizeof(float));
    cudaMemcpy(d_query, h_query, num_features * sizeof(float), cudaMemcpyHostToDevice);

    // Double-buffered device memory: one buffer per stream slot
    float* d_historical_buf[NUM_STREAMS];
    float* d_distances_buf[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaMalloc(&d_historical_buf[s], (size_t)batch_size * num_features * sizeof(float));
        cudaMalloc(&d_distances_buf[s], batch_size * sizeof(float));
    }

    int num_batches = (num_rows + batch_size - 1) / batch_size;

    // Separate event pairs so transfer time and kernel time are logged
    // independently, per the assignment's requirement.
    cudaEvent_t xfer_start[NUM_STREAMS], xfer_stop[NUM_STREAMS];
    cudaEvent_t kern_start[NUM_STREAMS], kern_stop[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaEventCreate(&xfer_start[s]); cudaEventCreate(&xfer_stop[s]);
        cudaEventCreate(&kern_start[s]); cudaEventCreate(&kern_stop[s]);
    }

    total_kernel_ms_out = 0.0f;
    total_transfer_ms_out = 0.0f;

    for (int b = 0; b < num_batches; b++) {
        int slot = b % NUM_STREAMS;
        cudaStream_t s = streams[slot];

        int row_start = b * batch_size;
        int rows_this_batch = std::min(batch_size, num_rows - row_start);
        size_t bytes_this_batch = (size_t)rows_this_batch * num_features * sizeof(float);

        // Async H2D copy of this batch on this stream's buffer
        cudaEventRecord(xfer_start[slot], s);
        cudaMemcpyAsync(
            d_historical_buf[slot],
            h_historical_pinned + (size_t)row_start * num_features,
            bytes_this_batch,
            cudaMemcpyHostToDevice,
            s
        );
        cudaEventRecord(xfer_stop[slot], s);

        // Tiled kernel launch on the SAME stream, right after its own
        // copy -> operations queued on stream B (slot 1) can run
        // concurrently with stream A (slot 0)'s copy/compute, giving
        // the overlap between batch N compute and batch N+1 transfer.
        int blocks = (rows_this_batch + TILE_ROWS - 1) / TILE_ROWS;
        size_t shmem_bytes = (num_features + (size_t)TILE_ROWS * num_features) * sizeof(float);

        cudaEventRecord(kern_start[slot], s);
        distance_kernel_tiled<<<blocks, TILE_ROWS, shmem_bytes, s>>>(
            d_query, d_historical_buf[slot], d_distances_buf[slot], rows_this_batch, num_features
        );
        cudaEventRecord(kern_stop[slot], s);

        // Async D2H copy of results back
        cudaMemcpyAsync(
            h_distances_out + row_start,
            d_distances_buf[slot],
            rows_this_batch * sizeof(float),
            cudaMemcpyDeviceToHost,
            s
        );
    }

    for (int s = 0; s < NUM_STREAMS; s++) cudaStreamSynchronize(streams[s]);

    // Accumulate logged times (synchronize each event pair now that all
    // streams are done, so cudaEventElapsedTime is safe to call)
    for (int s = 0; s < NUM_STREAMS; s++) {
        float t_ms;
        cudaEventElapsedTime(&t_ms, xfer_start[s], xfer_stop[s]);
        total_transfer_ms_out += t_ms;
        cudaEventElapsedTime(&t_ms, kern_start[s], kern_stop[s]);
        total_kernel_ms_out += t_ms;
    }

    printf("[hybrid] Batches: %d | Batch size: %d\n", num_batches, batch_size);
    printf("[hybrid] Total async transfer time (logged): %.4f ms\n", total_transfer_ms_out);
    printf("[hybrid] Total kernel time (logged): %.4f ms\n", total_kernel_ms_out);

    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaEventDestroy(xfer_start[s]); cudaEventDestroy(xfer_stop[s]);
        cudaEventDestroy(kern_start[s]); cudaEventDestroy(kern_stop[s]);
        cudaFree(d_historical_buf[s]);
        cudaFree(d_distances_buf[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_query);
}
