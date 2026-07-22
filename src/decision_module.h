/*
 * CS-387L - Parallel & Distributed Computing Lab
 * Parallel Fraud Detection System
 *
 * Group Members:
 *   Javairia Waseem   - 231400097
 *   Kashaf Ul Hassan  - 231400102
 *   Ajwa Imran        - 231400105
 */

#ifndef DECISION_MODULE_H
#define DECISION_MODULE_H

#include <vector>

// Given distances to all historical points, find the K nearest and vote.
//
// useReduction:
//   false -> voting uses #pragma omp atomic on a shared counts[2] array
//   true  -> voting uses OpenMP reduction(+:...) with thread-local counts
//   (both are required by the spec so you can benchmark/compare them)
//
// selection_ms_out / voting_ms_out:
//   optional output params (omp_get_wtime-based) so callers can log the
//   "OpenMP selection/voting time" required in the profiling table.
//   Pass nullptr if you don't need them.
//
// Returns predicted label: 0 = Non-Fraud, 1 = Fraud. Ties resolve to Fraud.
int classify_by_knn(
    const std::vector<float>& distances,
    const std::vector<int>& labels,
    int K,
    bool useReduction = false,
    double* selection_ms_out = nullptr,
    double* voting_ms_out = nullptr
);

// Single-threaded reference baseline: full std::partial_sort selection +
// a plain sequential vote count, no OpenMP anywhere. This is T1 -- the
// serial time your report's S = Tserial / Tparallel speedup figures and
// Work-Span analysis are measured against. Must produce the SAME
// prediction as classify_by_knn for the same inputs (use this to validate
// correctness of the parallel version too).
int classify_by_knn_serial(
    const std::vector<float>& distances,
    const std::vector<int>& labels,
    int K,
    double* selection_ms_out = nullptr,
    double* voting_ms_out = nullptr
);

#endif
