#include <iostream>
#include <vector>
#include "preprocessing.h"
#include "decision_module.h"
#include "distance_calculation.cuh"
#include "timer.h"
#include <omp.h>

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
    std::cout << "[main] Load time: " << load_time << " ms" << std::endl;

    // Module 2 + 3 would run here once implemented

    std::cout << "[main] Build pipeline verified successfully (stub run)." << std::endl;
    return 0;
}