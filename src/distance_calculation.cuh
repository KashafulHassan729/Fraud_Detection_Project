/*
 * CS-387L - Parallel & Distributed Computing Lab
 * Parallel Fraud Detection System
 *
 * Group Members:
 *   Javairia Waseem   - 231400097
 *   Kashaf Ul Hassan  - 231400102
 *   Ajwa Imran        - 231400105
 */

#ifndef DISTANCE_CALCULATION_CUH
#define DISTANCE_CALCULATION_CUH

void compute_distances_cuda(
    const float* h_query,
    const float* h_historical,
    float* h_distances_out,
    int num_rows,
    int num_features
);


void compute_distances_tiled(
    const float* h_query,
    const float* h_historical,
    float* h_distances_out,
    int num_rows,
    int num_features
);

void benchmark_naive_vs_tiled(
    const float* h_query,
    const float* h_historical,
    int num_rows,
    int num_features,
    float& naive_ms_out,
    float& tiled_ms_out
);


void benchmark_pinned_vs_pageable(
    const float* h_historical_source,
    int num_rows,
    int num_features,
    float& pageable_ms_out,
    float& pinned_ms_out
);


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
);

#endif

