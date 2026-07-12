#ifndef DISTANCE_CALCULATION_CUH
#define DISTANCE_CALCULATION_CUH

// ===== Module 2: naive baseline (1 thread = 1 distance) =====
void compute_distances_cuda(
    const float* h_query,
    const float* h_historical,
    float* h_distances_out,
    int num_rows,
    int num_features
);

// ===== Module 2: tiled shared-memory kernel =====
void compute_distances_tiled(
    const float* h_query,
    const float* h_historical,
    float* h_distances_out,
    int num_rows,
    int num_features
);

// Runs naive then tiled on the same data, times each with cudaEvent_t,
// prints + returns both kernel times (ms) via out params.
void benchmark_naive_vs_tiled(
    const float* h_query,
    const float* h_historical,
    int num_rows,
    int num_features,
    float& naive_ms_out,
    float& tiled_ms_out
);

// Times a Host->Device copy of the historical array using pageable
// (malloc) memory vs pinned (cudaHostAlloc) memory.
void benchmark_pinned_vs_pageable(
    const float* h_historical_source,
    int num_rows,
    int num_features,
    float& pageable_ms_out,
    float& pinned_ms_out
);

// ===== Module 4: hybrid batched streaming pipeline =====
// Splits num_rows historical records into batches, uses 2 non-default
// CUDA streams so GPU compute on batch N overlaps with async H2D
// transfer of batch N+1. Requires h_historical to be pinned memory
// for the async copies to actually run concurrently with compute.
void compute_distances_hybrid_streamed(
    const float* h_query,
    const float* h_historical_pinned,
    float* h_distances_out,
    int num_rows,
    int num_features,
    int batch_size,
    float& total_kernel_ms_out,
    float& total_transfer_ms_out
);

#endif
