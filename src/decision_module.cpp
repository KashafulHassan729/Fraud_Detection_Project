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

// -----------------------------------------------------------------------
// PARALLEL: Nearest Neighbor Extraction (Partial Selection)
// -----------------------------------------------------------------------
// Full sort is disallowed by the spec. Instead: each thread keeps its own
// bounded max-heap of size K over its chunk of the distance array. The
// heap's top() is always the CURRENT WORST (largest-distance) neighbor kept
// so far, so a new candidate only needs O(log K) work to decide whether it
// belongs in the top-K. Total cost is O(n log K) instead of O(n log n).
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

    std::partial_sort(merged.begin(),
                       merged.begin() + std::min((size_t)K, merged.size()),
                       merged.end(), byDistAsc);
    if ((int)merged.size() > K) merged.resize(K);
    return merged;
}

// -----------------------------------------------------------------------
// PARALLEL: Majority Voting Framework (atomic vs reduction)
// -----------------------------------------------------------------------
// Race condition being guarded against: without synchronization, two
// threads could both read counts[label], both increment locally, and both
// write back the same incremented value -- one increment is silently lost
// (a classic read-modify-write race). Document this explicitly in your report.
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

// -----------------------------------------------------------------------
// SERIAL BASELINE (T1 reference) -- no OpenMP anywhere in this function.
// -----------------------------------------------------------------------
// Selection: plain std::partial_sort over the whole distance array. Still
// avoids a FULL sort (partial_sort is O(n log K) same as the parallel
// version's asymptotic complexity) -- the serial/parallel comparison here
// isolates the effect of THREADING, not a different algorithm, which is
// what makes the resulting speedup number meaningful.
// Voting: a single sequential counts[2] increment loop, no synchronization
// needed since there's only one thread.
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
