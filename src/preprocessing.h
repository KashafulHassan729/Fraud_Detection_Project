#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <string>
using namespace std;

void benchmark_thread_counts(const string& filepath, int num_features);

void load_and_normalize_csv(
    const string& filepath,
    vector<float>& features_out,
    vector<int>& labels_out,
    int& num_rows,
    int num_features
);

void load_and_normalize_csv_dynamic(
    const string& filepath,
    vector<float>& features_out,
    vector<int>& labels_out,
    int& num_rows,
    int num_features
);

// Single-threaded reference baseline -- no OpenMP anywhere in this
// function (parsing, min/max scan, and normalization all run on one
// thread). This is T1, the serial time your report's speedup figures
// (S = Tserial / Tparallel) and Work-Span analysis for Module 1 are
// measured against. Produces identical output to the parallel versions
// for the same input file (use to sanity-check correctness too).
void load_and_normalize_csv_serial(
    const string& filepath,
    vector<float>& features_out,
    vector<int>& labels_out,
    int& num_rows,
    int num_features
);

#endif
