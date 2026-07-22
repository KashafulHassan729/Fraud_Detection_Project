/*
 * CS-387L - Parallel & Distributed Computing Lab
 * Parallel Fraud Detection System
 *
 * Group Members:
 *   Javairia Waseem   - 231400097
 *   Kashaf Ul Hassan  - 231400102
 *   Ajwa Imran        - 231400105
 */

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

#endif