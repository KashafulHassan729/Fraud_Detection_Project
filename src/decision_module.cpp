#include "decision_module.h"

#include <omp.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>

struct Neighbor {
    float dist;
    int   idx;
};

static bool byDistAsc(const Neighbor& a, const Neighbor& b) {
    return a.dist < b.dist;
}
static std::vector<Neighbor> selectTopK_partial(const std::vector<float>& distances, int K) {
    int n = (int)distances.size();
    if (K <= 0 || n <= 0) return {};
    if (K > n) K = n;

    int maxThreads = omp_get_max_threads();
    std::vector<std::vector<Neighbor>> threadLocalTopK(maxThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto cmp = [](const Neighbor& a, const Neighbor& b) { return a.dist < b.dist; };
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp)> localHeap(cmp);

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            Neighbor cand{distances[i], i};
            if ((int)localHeap.size() < K) {
                localHeap.push(cand);
            } else if (cand.dist < localHeap.top().dist) {
                localHeap.pop();
                localHeap.push(cand);
            }
        }

        std::vector<Neighbor>& out = threadLocalTopK[tid];
        out.reserve(localHeap.size());
        while (!localHeap.empty()) {
            out.push_back(localHeap.top());
            localHeap.pop();
        }
    }

    std::vector<Neighbor> merged;
    merged.reserve((size_t)maxThreads * K);
    for (auto& v : threadLocalTopK) merged.insert(merged.end(), v.begin(), v.end());

    std::partial_sort(merged.begin(),merged.begin() + std::min((size_t)K, merged.size()),merged.end(), byDistAsc);
    if ((int)merged.size() > K) merged.resize(K);
    return merged;
}
static int voteAtomic(const std::vector<Neighbor>& neighbors, const std::vector<int>& labels) {
    int counts[2] = {0, 0};
    int K = (int)neighbors.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < K; ++i) {
        int label = labels[neighbors[i].idx];
        #pragma omp atomic
        counts[label]++;
    }
    return (counts[1] >= counts[0]) ? 1 : 0; // tie -> Fraud (conservative default)
}

static int voteReduction(const std::vector<Neighbor>& neighbors, const std::vector<int>& labels) {
    int K = (int)neighbors.size();
    int fraudVotes = 0, nonFraudVotes = 0;

    #pragma omp parallel for schedule(static) reduction(+:fraudVotes, nonFraudVotes)
    for (int i = 0; i < K; ++i) {
        int label = labels[neighbors[i].idx];
        if (label == 1) fraudVotes++;
        else            nonFraudVotes++;
    }
    return (fraudVotes >= nonFraudVotes) ? 1 : 0;
}

int classify_by_knn(
    const std::vector<float>& distances,
    const std::vector<int>& labels,
    int K,
    bool useReduction,
    double* selection_ms_out,
    double* voting_ms_out
) {
    double t0 = omp_get_wtime();
    std::vector<Neighbor> topK = selectTopK_partial(distances, K);
    double t1 = omp_get_wtime();

    int prediction = useReduction ? voteReduction(topK, labels)
                                   : voteAtomic(topK, labels);
    double t2 = omp_get_wtime();

    if (selection_ms_out) *selection_ms_out = (t1 - t0) * 1000.0;
    if (voting_ms_out)    *voting_ms_out    = (t2 - t1) * 1000.0;

    return prediction;
}
static int voteSerial(const std::vector<Neighbor>& neighbors, const std::vector<int>& labels) {
    int counts[2] = {0, 0};
    for (const auto& nb : neighbors) {
        counts[labels[nb.idx]]++;
    }
    return (counts[1] >= counts[0]) ? 1 : 0;
}

int classify_by_knn_serial(
    const std::vector<float>& distances,
    const std::vector<int>& labels,
    int K,
    double* selection_ms_out,
    double* voting_ms_out
) {
    int n = (int)distances.size();
    if (K > n) K = n;

    using clock = std::chrono::high_resolution_clock;

    auto t0 = clock::now();
    std::vector<Neighbor> all(n);
    for (int i = 0; i < n; ++i) all[i] = {distances[i], i};
    std::partial_sort(all.begin(), all.begin() + K, all.end(), byDistAsc);
    std::vector<Neighbor> topK(all.begin(), all.begin() + K);
    auto t1 = clock::now();

    int prediction = voteSerial(topK, labels);
    auto t2 = clock::now();

    if (selection_ms_out) *selection_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (voting_ms_out)    *voting_ms_out    = std::chrono::duration<double, std::milli>(t2 - t1).count();

    return prediction;
}
