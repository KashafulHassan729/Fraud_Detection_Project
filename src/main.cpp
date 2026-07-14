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
    int K = 5;
    if (argc > 1) {
        K = std::atoi(argv[1]);
    }

    std::cout << "=== Fraud Detection System ===" << std::endl;
    std::cout << "K = " << K << std::endl;

    Timer t;
    t.start();

    std::vector<float> features;
    std::vector<int> labels;
    int num_rows = 0;
    int num_features = 16;

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

    vector<float> query(features.begin(), features.begin() + num_features);

    printf("\n===== MODULE 2: NAIVE vs TILED CUDA KERNEL =====\n");
    float naive_ms = 0.0f, tiled_ms = 0.0f;
    benchmark_naive_vs_tiled(query.data(), features.data(), num_rows, num_features, naive_ms, tiled_ms);

    printf("\n===== MODULE 2: PINNED vs PAGEABLE HOST MEMORY =====\n");
    float pageable_ms = 0.0f, pinned_ms = 0.0f;
    benchmark_pinned_vs_pageable(features.data(), num_rows, num_features, pageable_ms, pinned_ms);

    printf("\n===== MODULE 4: HYBRID BATCHED STREAMING PIPELINE =====\n");
    vector<float> hybrid_distances(num_rows);
    int batch_size = 8000;
    float hybrid_kernel_ms = 0.0f, hybrid_transfer_ms = 0.0f, hybrid_cpu_prep_ms = 0.0f;
    compute_distances_hybrid_streamed(
        query.data(), features.data(), hybrid_distances.data(),
        num_rows, num_features, batch_size,
        hybrid_kernel_ms, hybrid_transfer_ms, hybrid_cpu_prep_ms
    );

    // ===== Module 3: Decision & Voting Engine =====
    // Runs serial (T1 baseline), atomic, and reduction for each K, and
    // logs speedup S = Tserial / Tparallel for each parallel mode --
    // required for the report's Work-Span / speedup analysis.
    printf("\n===== MODULE 3: DECISION & VOTING (K sweep) =====\n");

    vector<int> K_values = {3, 5, 7, 11};
    FILE* module3_csv = fopen("output/decision_voting_report.csv", "w");
    if (module3_csv) {
        fprintf(module3_csv, "K,mode,predicted_label,selection_ms,voting_ms,speedup_selection_vs_serial\n");
    } else {
        fprintf(stderr, "[main] WARNING: could not open output/decision_voting_report.csv for logging\n");
    }

    for (int Kval : K_values) {
        // --- Serial baseline (T1) ---
        double serial_sel_ms = 0.0, serial_vote_ms = 0.0;
        int serial_pred = classify_by_knn_serial(hybrid_distances, labels, Kval, &serial_sel_ms, &serial_vote_ms);

        printf("K=%-2d mode=%-9s predicted=%-9s selection=%.4fms voting=%.4fms\n",
               Kval, "serial", serial_pred ? "FRAUD" : "NON-FRAUD", serial_sel_ms, serial_vote_ms);
        if (module3_csv) {
            fprintf(module3_csv, "%d,serial,%d,%.4f,%.4f,%.4f\n",
                    Kval, serial_pred, serial_sel_ms, serial_vote_ms, 1.0);
        }

        // --- Parallel: atomic and reduction, each compared back to serial ---
        for (int mode = 0; mode < 2; ++mode) {
            bool useReduction = (mode == 1);
            double sel_ms = 0.0, vote_ms = 0.0;
            int pred = classify_by_knn(hybrid_distances, labels, Kval, useReduction, &sel_ms, &vote_ms);
            double speedup = (sel_ms > 0.0) ? (serial_sel_ms / sel_ms) : 0.0;

            printf("K=%-2d mode=%-9s predicted=%-9s selection=%.4fms voting=%.4fms speedup=%.2fx\n",
                   Kval, useReduction ? "reduction" : "atomic",
                   pred ? "FRAUD" : "NON-FRAUD", sel_ms, vote_ms, speedup);

            if (module3_csv) {
                fprintf(module3_csv, "%d,%s,%d,%.4f,%.4f,%.4f\n",
                        Kval, useReduction ? "reduction" : "atomic", pred, sel_ms, vote_ms, speedup);
            }
        }
    }
    if (module3_csv) fclose(module3_csv);

    // ===== Hardcoded validation queries (spec requirement: 3+ test cases) =====
    printf("\n===== VALIDATION QUERIES =====\n");
    struct TestQuery { const char* name; vector<float> vec; };
    vector<TestQuery> testQueries = {
        {"Low-amount, typical pattern",      vector<float>(num_features, 0.05f)},
        {"High-amount, outlier pattern",     vector<float>(num_features, 0.95f)},
        {"Mid-range, borderline pattern",    vector<float>(num_features, 0.50f)}
    };

    for (auto& tq : testQueries) {
        vector<float> test_distances(num_rows);
        compute_distances_tiled(tq.vec.data(), features.data(), test_distances.data(), num_rows, num_features);
        int pred = classify_by_knn(test_distances, labels, K);
        printf("[%-30s] predicted = %s\n", tq.name, pred ? "FRAUD" : "NON-FRAUD");
    }

    // ===== Automated output routing to performance_report.csv =====
    FILE* report = fopen("output/performance_report.csv", "a");
    if (report) {
        fseek(report, 0, SEEK_END);
        if (ftell(report) == 0) {
            fprintf(report, "K,num_rows,num_features,static_sec,dynamic_sec,naive_kernel_ms,tiled_kernel_ms,pageable_copy_ms,pinned_copy_ms,hybrid_kernel_ms,hybrid_transfer_ms,hybrid_cpu_prep_ms\n");
        }
        fprintf(report, "%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            K, num_rows, num_features, t2 - t1, t4 - t3,
            naive_ms, tiled_ms, pageable_ms, pinned_ms,
            hybrid_kernel_ms, hybrid_transfer_ms, hybrid_cpu_prep_ms);
        fclose(report);
        printf("\n[main] Logged this run to output/performance_report.csv\n");
    } else {
        fprintf(stderr, "[main] WARNING: could not open output/performance_report.csv for logging\n");
    }

    std::cout << "[main] Module 1 + Module 2 + Module 3 + Module 4 run complete." << std::endl;
    return 0;
}
