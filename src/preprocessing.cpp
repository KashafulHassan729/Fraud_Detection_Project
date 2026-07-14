#include "preprocessing.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <string>
#include <limits>
#include <omp.h>
using namespace std;

void benchmark_thread_counts(
    const string& filepath,
    int num_features
) {
    FILE* csv = fopen("output/preprocessing_report.csv", "w");
    if (csv) {
        fprintf(csv, "mode,threads,time_sec,speedup_vs_serial\n");
    } else {
        fprintf(stderr, "[preprocessing] WARNING: could not open output/preprocessing_report.csv for logging\n");
    }

    // --- Serial baseline (T1), run once, thread count is not meaningful here ---
    printf("\n===== SERIAL BASELINE (single-threaded) =====\n");
    vector<float> features_serial;
    vector<int> labels_serial;
    int num_rows_serial = 0;

    double t_serial_start = omp_get_wtime();
    load_and_normalize_csv_serial(filepath, features_serial, labels_serial, num_rows_serial, num_features);
    double t_serial_end = omp_get_wtime();
    double serial_time = t_serial_end - t_serial_start;

    printf("-> Serial | Total time: %.4f sec\n", serial_time);
    if (csv) {
        fprintf(csv, "serial,1,%.4f,%.4f\n", serial_time, 1.0);
    }

    int thread_counts[4] = {1,2,4,8};
    printf("\n===== THREAD COUNT BENCHMARK (schedule static) =====\n");
    for(int i = 0 ; i < 4; i++){
        int tc = thread_counts[i];
        omp_set_num_threads(tc);

        vector<float> features;
        vector<int> labels;
        int num_rows = 0;

        double t_start = omp_get_wtime();
        load_and_normalize_csv(filepath, features, labels, num_rows,num_features);
        double t_end = omp_get_wtime();
        double elapsed = t_end - t_start;
        double speedup = (elapsed > 0.0) ? (serial_time / elapsed) : 0.0;

        printf("-> Threads %d | Total time: %.4f sec | Speedup vs serial: %.2fx\n", tc, elapsed, speedup);
        if (csv) {
            fprintf(csv, "static,%d,%.4f,%.4f\n", tc, elapsed, speedup);
        }
    }

    printf("\n===== THREAD COUNT BENCHMARK (schedule dynamic) =====\n");
    for(int i = 0 ; i < 4; i++){
        int tc = thread_counts[i];
        omp_set_num_threads(tc);

        vector<float> features;
        vector<int> labels;
        int num_rows = 0;

        double t_start = omp_get_wtime();
        load_and_normalize_csv_dynamic(filepath,features,labels,num_rows, num_features);
        double t_end = omp_get_wtime();
        double elapsed = t_end - t_start;
        double speedup = (elapsed > 0.0) ? (serial_time / elapsed) : 0.0;

        printf("-> Threads: %d | Total time: %.4f sec | Speedup vs serial: %.2fx\n", tc, elapsed, speedup);
        if (csv) {
            fprintf(csv, "dynamic,%d,%.4f,%.4f\n", tc, elapsed, speedup);
        }
    }

    if (csv) {
        fclose(csv);
        printf("\n[preprocessing] Logged serial/static/dynamic comparison to output/preprocessing_report.csv\n");
    }
}

void load_and_normalize_csv_dynamic(
    const string& filepath,
    vector<float>& features_out,
    vector<int>& labels_out,
    int& num_rows,
    int num_features
) {
    ifstream file(filepath);
    if(!file.is_open()){
        fprintf(stderr,"[preprocessing-dynamic] ERROR: could not open %s\n",filepath.c_str());
        num_rows = 0;
        return;
    }

    string line;
    getline(file,line);

    double t_read_start = omp_get_wtime();

    vector<string> raw_lines;
    while(getline(file,line)){
        if(!line.empty()) raw_lines.push_back(line);
    }

    file.close();

    int row_count = (int)raw_lines.size();
    features_out.resize((size_t)row_count * num_features);
    labels_out.resize(row_count);

    double t_read_end = omp_get_wtime();
    printf("[preprocessing-dynamic] File read (serial): %.4f sec, %d rows\n", t_read_end - t_read_start, row_count);

    double t_parse_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for(int row = 0 ; row < row_count; row++){
        stringstream ss(raw_lines[row]);
        string cell;

        for(int col = 0; col < num_features; col++){
            getline(ss,cell,',');
            features_out[(size_t)row * num_features + col] = stof(cell);
        }

        getline(ss,cell,',');
        labels_out[row] = stoi(cell);
    }

    double t_parse_end = omp_get_wtime();
    num_rows = row_count;

    printf("[preprocessing-dynamic] Parsing (parallel, dynamic): %.4f sec, using %d threads\n", t_parse_end - t_parse_start, omp_get_max_threads());

    double t_minmax_start  = omp_get_wtime();
    vector<float> col_min(num_features, 1e9f);
    vector<float> col_max(num_features, -1e9f);

    for(int c = 0; c < num_features; c++){
        float local_min = 1e9f;
        float local_max = -1e9f;

        #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) schedule(dynamic)
        for(int r = 0 ; r <num_rows; r++){
            float val = features_out[(size_t)r * num_features + c];
            if(val < local_min) local_min = val;
            if(val > local_max) local_max = val;
        }

        col_min[c] = local_min;
        col_max[c] = local_max;
    }

    double t_minmax_end = omp_get_wtime();
    printf("[preprocessing-dynamic] Min/max scan (parallel): %.4f sec\n", t_minmax_end - t_minmax_start);

    double t_norm_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for(int r = 0 ; r < num_rows; r++){
        for(int c = 0 ; c < num_features; c++){
            size_t idx = (size_t)r * num_features + c;
            float range = col_max[c] - col_min[c];
            if(range > 0.0f){
                features_out[idx] = (features_out[idx] -  col_min[c]) / range;
            } else {
                features_out[idx] = 0.0f;
            }
        }
    }

    double t_norm_end = omp_get_wtime();
    printf("[preprocessing-dynamic] Normalization (parallel): %.4f sec\n", t_norm_end - t_norm_start);
    printf("[preprocessing-dynamic] Loaded %d rows, %d features each.\n", num_rows, num_features);
    printf("[preprocessing-dynamic] Normalization complete!\n");
}

void load_and_normalize_csv(
    const string& filepath,
    vector<float>& features_out,
    vector<int>& labels_out,
    int& num_rows,
    int num_features
) {
    ifstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "[preprocessing] ERROR: could not open %s\n", filepath.c_str());
        num_rows = 0;
        return;
    }

    string line;
    getline(file, line); 

    double t_read_start = omp_get_wtime();

    vector<string> raw_lines;
    while (getline(file, line)) {
        if (!line.empty()) raw_lines.push_back(line);
    }
    file.close();

    int row_count = (int)raw_lines.size();
    features_out.resize((size_t)row_count * num_features);
    labels_out.resize(row_count);

    double t_read_end = omp_get_wtime();
    printf("[preprocessing] File read (serial): %.4f sec, %d rows\n", t_read_end - t_read_start, row_count);

   double t_parse_start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < row_count; row++) {
        stringstream ss(raw_lines[row]);
        string cell;

        for (int col = 0; col < num_features; col++) {
            getline(ss, cell, ',');
            features_out[(size_t)row * num_features + col] = stof(cell);
        }

        getline(ss, cell, ',');
        labels_out[row] = stoi(cell);
    }

    double t_parse_end = omp_get_wtime();
    num_rows = row_count;

    printf("[preprocessing] Parsing (parallel, static): %.4f sec, using %d threads\n",
           t_parse_end - t_parse_start, omp_get_max_threads());

    double t_minmax_start = omp_get_wtime();

    vector<float> col_min(num_features, 1e9f);
    vector<float> col_max(num_features, -1e9f);

    for (int c = 0; c < num_features; c++) {
        float local_min = 1e9f;
        float local_max = -1e9f;

        #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) schedule(static)
        for (int r = 0; r < num_rows; r++) {
            float val = features_out[(size_t)r * num_features + c];
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }

        col_min[c] = local_min;
        col_max[c] = local_max;
    }

    double t_minmax_end = omp_get_wtime();
    printf("[preprocessing] Min/max scan (parallel): %.4f sec\n", t_minmax_end - t_minmax_start);

    double t_norm_start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_features; c++) {
            size_t idx = (size_t)r * num_features + c;
            float range = col_max[c] - col_min[c];
            if (range > 0.0f) {
                features_out[idx] = (features_out[idx] - col_min[c]) / range;
            } else {
                features_out[idx] = 0.0f;
            }
        }
    }

    double t_norm_end = omp_get_wtime();
    printf("[preprocessing] Normalization (parallel): %.4f sec\n", t_norm_end - t_norm_start);

    printf("[preprocessing] Loaded %d rows, %d features each.\n", num_rows, num_features);
    printf("[preprocessing] Normalization complete!\n");
}

// ---------------------------------------------------------------------------
// SERIAL BASELINE (T1 reference) -- identical algorithm to load_and_normalize_csv
// above, but with every #pragma omp removed and every loop run on a single
// thread. This isolates the effect of THREADING alone (same big-O work,
// same normalization formula), which is what makes the resulting
// S = Tserial / Tparallel speedup number meaningful for your report.
// ---------------------------------------------------------------------------
void load_and_normalize_csv_serial(
    const string& filepath,
    vector<float>& features_out,
    vector<int>& labels_out,
    int& num_rows,
    int num_features
) {
    ifstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "[preprocessing-serial] ERROR: could not open %s\n", filepath.c_str());
        num_rows = 0;
        return;
    }

    string line;
    getline(file, line); // skip header

    double t_read_start = omp_get_wtime();

    vector<string> raw_lines;
    while (getline(file, line)) {
        if (!line.empty()) raw_lines.push_back(line);
    }
    file.close();

    int row_count = (int)raw_lines.size();
    features_out.resize((size_t)row_count * num_features);
    labels_out.resize(row_count);

    double t_read_end = omp_get_wtime();
    printf("[preprocessing-serial] File read: %.4f sec, %d rows\n", t_read_end - t_read_start, row_count);

    // --- Parsing: plain sequential loop, no #pragma omp ---
    double t_parse_start = omp_get_wtime();

    for (int row = 0; row < row_count; row++) {
        stringstream ss(raw_lines[row]);
        string cell;

        for (int col = 0; col < num_features; col++) {
            getline(ss, cell, ',');
            features_out[(size_t)row * num_features + col] = stof(cell);
        }

        getline(ss, cell, ',');
        labels_out[row] = stoi(cell);
    }

    double t_parse_end = omp_get_wtime();
    num_rows = row_count;

    printf("[preprocessing-serial] Parsing: %.4f sec, single thread\n", t_parse_end - t_parse_start);

    // --- Min/max scan: plain sequential loop, no reduction needed ---
    double t_minmax_start = omp_get_wtime();

    vector<float> col_min(num_features, 1e9f);
    vector<float> col_max(num_features, -1e9f);

    for (int c = 0; c < num_features; c++) {
        float mn = 1e9f;
        float mx = -1e9f;

        for (int r = 0; r < num_rows; r++) {
            float val = features_out[(size_t)r * num_features + c];
            if (val < mn) mn = val;
            if (val > mx) mx = val;
        }

        col_min[c] = mn;
        col_max[c] = mx;
    }

    double t_minmax_end = omp_get_wtime();
    printf("[preprocessing-serial] Min/max scan: %.4f sec\n", t_minmax_end - t_minmax_start);

    // --- Normalization: plain sequential loop ---
    double t_norm_start = omp_get_wtime();

    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_features; c++) {
            size_t idx = (size_t)r * num_features + c;
            float range = col_max[c] - col_min[c];
            if (range > 0.0f) {
                features_out[idx] = (features_out[idx] - col_min[c]) / range;
            } else {
                features_out[idx] = 0.0f;
            }
        }
    }

    double t_norm_end = omp_get_wtime();
    printf("[preprocessing-serial] Normalization: %.4f sec\n", t_norm_end - t_norm_start);

    printf("[preprocessing-serial] Loaded %d rows, %d features each.\n", num_rows, num_features);
    printf("[preprocessing-serial] Normalization complete!\n");
}
