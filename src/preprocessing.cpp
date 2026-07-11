#include "preprocessing.h"
#include <iostream>
#include <fstream>
#include <sstream>

void load_and_normalize_csv(
    const std::string& filepath,
    std::vector<float>& features_out,
    std::vector<int>& labels_out,
    int& num_rows,
    int num_features
) {
    std::cout << "[preprocessing] STUB: would load " << filepath << std::endl;

    // TODO: parse CSV, fill features_out / labels_out, normalize
    // TODO: parallelize the row-parsing loop with #pragma omp parallel for

    num_rows = 0; // placeholder until implemented
}