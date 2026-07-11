#include "preprocessing.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <string>
#include <limits>
#include <omp.h>
using namespace std;

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

    double_t_read_start = omp_get_wtime();

    vector<string> raw_lines;
    while(getline(file,line){
        if(!line.empty()) raw_lines.push_back(line);
    })

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
    printf('[preprocessing-dynamic] Min/max scan (parallel): %.4f sec\n', t_minmax_end - t_minmax_start);

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
    printf("[preproccesing-dynamic] Normalization (parallel): %.4f sec\n", t_norm_end - t_norm_start);
    printf("[preprocessing-dynamic] Loaded %d rows, %d features each. \n", num_rows, num_features);
    printf("[preprocessing-dynamic] Normalization complete! \n")


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