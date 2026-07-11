#include "preprocessing.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
using namespace std;

void load_and_normalize_csv(
    const string& filepath,
    vector<float>& features_out,
    vector<int>& labels_out,
    int& num_rows,
    int num_features
) {
    ifstream file(filepath):
    if(!file.is_open()){
        fprintf(stderr, "[preprocessing] ERROR: could not open %s\n", filepath.c_str());
        num_rows = 0;
        return;
    }

    string line;

    getline(file,line);

    streampos data_start = file.tellg();
    int row_count = 0;
    while(getline(file,line)){
        if(!line.empty()) row_count++;
    }

    features_out.resize((size_t)row_count * num_features);
    labels_out.resize(row_count);

    file.clear();
    file.seekg(data_start);

    int row = 0;
    while (getline(file , line)){
        if(line.empty()) continue;

        stringstream ss(line);
        string cell;

        for(int col = 0 ; col < num_features; col++){
            getline(ss,cel,",");
            features_out[(size_t)row * num_features + col] = stof(cell);
        }

        getline(ss,cell,',');
        labels_out[row] = stoi(cell);

        row++;
    }

    file.close();
    num_rows = row;

    printf("[preprocessing] Loaded %d rows, %d features each. \n", num_rows, num_features);

}