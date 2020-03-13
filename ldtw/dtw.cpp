#include "dtw.h"
#include <cmath>
#include <algorithm>
using namespace std;


vector<pair<int, int> > AlignmentPath(vector<vector<double> > &alignment_matrix,
                                      pair<int, int> argmax) {
    vector<pair<int, int> > alignment_path;
    pair<int, int> actual_index = argmax;
    while (1) {
        alignment_path.push_back(actual_index);
        int next_x = -1, next_y = -1;
        double max_val = 0;
        if (alignment_matrix[actual_index.first - 1][actual_index.second] > max_val) {
            next_x = actual_index.first - 1;
            next_y = actual_index.second;
            max_val = alignment_matrix[actual_index.first - 1][actual_index.second];
        }
        if (alignment_matrix[actual_index.first][actual_index.second - 1] > max_val) {
            next_x = actual_index.first;
            next_y = actual_index.second - 1;
        }
        if ((next_x == -1) || (next_y == -1)) break;
        actual_index = {next_x, next_y};
    }
    reverse(alignment_path.begin(), alignment_path.end());
    return alignment_path;
}

tuple<double, int, int, int, int> LikelihoodAlignment(const vector<double> &signal1,
                           const vector<double> &signal2,
                           const vector<double> &scoring_scheme,
                           const int bucket_size,
                           const double delta) {
    int N = signal1.size();
    int M = signal2.size();
    vector< vector<double> > A(N + 1, vector<double>(M + 1));
    double ans = -1;
    pair<int, int> act = {1, 1};
    for (int i = 0; i <= N; i++) A[i][0] = 0;
    for (int i = 0; i <= M; i++) A[0][i] = 0;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            double dist = min(abs(signal1[i - 1] - signal2[j - 1]), 5.0);
            int bucket_idx = dist*bucket_size;
            double cost = scoring_scheme[bucket_idx] - delta;
            A[i][j] = max(.0, cost + max(A[i][j - 1], A[i - 1][j]));
            if (A[i][j] > ans) {
                ans = A[i][j];
                act = {i, j};
            }
        }
    }
    auto alignment_path = AlignmentPath(A, act);
    return {ans,
            alignment_path.front().second, alignment_path.back().second,
            alignment_path.front().first, alignment_path.back().first,
    };
}
