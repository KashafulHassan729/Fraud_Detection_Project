#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include "preprocessing.h"
#include "decision_module.h"
#include "distance_calculation.cuh"
#include "timer.h"
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
    int K = 5; // default per spec
    if (argc > 1) {
        K = std::atoi(argv[1]);
    }

    std::cout << "=== Fraud Detection System ===" << std::endl;
    std::cout << "K = " << K << std::endl;

    Timer t;
    t.start();

    // Module 1: load + normalize (stub for now)
    std::vector<float> features;
    std::vector<int> labels;
    int num_rows = 0;
    int num_features = 16; // Amount + V1..V15

    printf("\n===== STATIC SCHEDULING RUN =====\n");
    double t1 = omp_get_wtime();
    load_and_normalize_csv("data/synthetic_credit_fraud.csv", features, labels, num_rows, num_features);
    double t2 = omp_get_wtime();
    printf("[main] TOTAL static time: %.4f sec\n", t2 - t1);

    vector<float> features_dyn;
    vector<int> labels_dyn;
    int num_rows_dyn = 0;

    printf("\n===== DYNAMIC SCHEDULING RUN =====\n");
    double t3 = omp_get_wtime();
    load_and_normalize_csv_dynamic("data/synthetic_credit_fraud.csv", features_dyn, labels_dyn, num_rows_dyn, num_features);
    double t4 = omp_get_wtime();
    printf("[main] TOTAL dynamic time: %.4f sec\n", t4 - t3);

    printf("\n===== COMPARISON =====\n");
    printf("Static  total: %.4f sec\n", t2 - t1);
    printf("Dynamic total: %.4f sec\n", t4 - t3);
    double load_time = t.stop_ms();
    printf("[main] Load time: %.4f ms\n", load_time);
    benchmark_thread_counts("data/synthetic_credit_fraud.csv", num_features);

    // ===== Module 2: naive vs tiled CUDA kernel benchmark =====
    // Use the first normalized row as a stand-in query vector (a real
    // "incoming transaction" would come from a separate stream/file).
    vector<float> query(features.begin(), features.begin() + num_features);

    printf("\n===== MODULE 2: NAIVE vs TILED CUDA KERNEL =====\n");
    float naive_ms = 0.0f, tiled_ms = 0.0f;
    benchmark_naive_vs_tiled(query.data(), features.data(), num_rows, num_features, naive_ms, tiled_ms);

    printf("\n===== MODULE 2: PINNED vs PAGEABLE HOST MEMORY =====\n");
    float pageable_ms = 0.0f, pinned_ms = 0.0f;
    benchmark_pinned_vs_pageable(features.data(), num_rows, num_features, pageable_ms, pinned_ms);

    // ===== Module 4: hybrid batched + streamed pipeline =====
    // cudaMemcpyAsync needs pinned host memory to actually run async,
    // so copy the normalized features into a pinned buffer first.
    printf("\n===== MODULE 4: HYBRID BATCHED STREAMING PIPELINE =====\n");
    float* features_pinned;
    size_t features_bytes = (size_t)num_rows * num_features * sizeof(float);
    cudaHostAlloc((void**)&features_pinned, features_bytes, cudaHostAllocDefault);
    memcpy(features_pinned, features.data(), features_bytes);

    vector<float> hybrid_distances(num_rows);
    int batch_size = 8000; // within the 5,000-10,000 range from the spec
    float hybrid_kernel_ms = 0.0f, hybrid_transfer_ms = 0.0f;
    compute_distances_hybrid_streamed(
        query.data(), features_pinned, hybrid_distances.data(),
        num_rows, num_features, batch_size,
        hybrid_kernel_ms, hybrid_transfer_ms
    );
    cudaFreeHost(features_pinned);

    // ===== Automated output routing: log everything to output/performance_report.csv =====
    FILE* report = fopen("output/performance_report.csv", "a");
    if (report) {
        // Write header once if file is new/empty
        fseek(report, 0, SEEK_END);
        if (ftell(report) == 0) {
            fprintf(report, "K,num_rows,num_features,static_sec,dynamic_sec,naive_kernel_ms,tiled_kernel_ms,pageable_copy_ms,pinned_copy_ms,hybrid_kernel_ms,hybrid_transfer_ms\n");
        }
        fprintf(report, "%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            K, num_rows, num_features, t2 - t1, t4 - t3,
            naive_ms, tiled_ms, pageable_ms, pinned_ms,
            hybrid_kernel_ms, hybrid_transfer_ms);
        fclose(report);
        printf("\n[main] Logged this run to output/performance_report.csv\n");
    } else {
        fprintf(stderr, "[main] WARNING: could not open output/performance_report.csv for logging\n");
    }

    std::cout << "[main] Module 1 + Module 2 + Module 4 run complete." << std::endl;
    return 0;
}
