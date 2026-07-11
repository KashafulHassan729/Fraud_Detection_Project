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



#endif