#ifndef DISTANCE_CALCULATION_CUH
#define DISTANCE_CALCULATION_CUH

// Launches CUDA kernel to compute Euclidean distances from query Q
// to all historical points P. Fills distances_out (host array).
void compute_distances_cuda(
    const float* h_query,        // single query vector, size = num_features
    const float* h_historical,   // flat array, size = num_rows * num_features
    float* h_distances_out,      // output, size = num_rows
    int num_rows,
    int num_features
);

#endif