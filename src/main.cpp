#include <iostream>
#include <vector>
#include "preprocessing.h"
#include "decision_module.h"
#include "distance_calculation.cuh"
#include "timer.h"

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

    load_and_normalize_csv("data/synthetic_credit_fraud.csv", features, labels, num_rows, num_features);

    double load_time = t.stop_ms();
    std::cout << "[main] Load time: " << load_time << " ms" << std::endl;

    // Module 2 + 3 would run here once implemented

    std::cout << "[main] Build pipeline verified successfully (stub run)." << std::endl;
    return 0;
}