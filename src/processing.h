#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <string>

// Loads CSV into flat arrays: features (row-major) and labels
void load_and_normalize_csv(
    const std::string& filepath,
    std::vector<float>& features_out,   // size = num_rows * num_features
    std::vector<int>& labels_out,       // size = num_rows
    int& num_rows,
    int num_features
);

#endif