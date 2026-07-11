#ifndef DECISION_MODULE_H
#define DECISION_MODULE_H

#include <vector>

// Given distances to all historical points, find the K nearest and vote
int classify_by_knn(
    const std::vector<float>& distances,
    const std::vector<int>& labels,
    int K
);

#endif