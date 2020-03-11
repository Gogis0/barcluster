#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include "dtw.h"
using namespace std;

struct Sentinel {
    int read_idx;
    int start;
    int end;

   string toString() {
       return to_string(read_idx) + ' ' + to_string(start) + ' ' + to_string(end);
   }
};


void AlignmentThread(vector<vector<double> > &D,
                    const vector<pair<int, int> > &indices,
                    const vector< vector<double> > &reads,
                    vector<vector<Sentinel> > &frontiers,
                    const vector<double> &scores,
                    const int bucket_size,
                    const int start, const int end) {
    for (int idx = start; idx < end; idx++) {
        int i = indices[idx].first, j = indices[idx].second;
        cout << "computing " << i << " " << j << endl;
        auto ans = LikelihoodAlignment(reads[i], reads[j], scores, bucket_size, 0.5);

        Sentinel sentinel_i = { .read_idx = j, .start = get<1>(ans), .end = get<2>(ans) };
        Sentinel sentinel_j = { .read_idx = i, .start = get<3>(ans), .end = get<4>(ans) };
        D[i][j] = D[j][i] = get<0>(ans);
        if (i != j) {
            frontiers[i].push_back(sentinel_i);
            frontiers[j].push_back(sentinel_j);
        }
    }
}


auto load_read(ifstream &file, int length) {
    vector<double> read(length);
    for (int i = 0; i < length; i++) {
        file >> read[i];
    }
    return read;
}

void WriteMatrix(ofstream &file,
                 vector<vector<Sentinel> > &frontiers,
                 const vector<vector<double> > &distance) {
    int N = distance.size();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < frontiers[i].size(); j++) {
            file << frontiers[i][j].toString() << endl;
        }
        for (int j = 0; j < N; j++) {
            file << distance[i][j];
            if (j < N - 1) file << ',';
        }
        file << endl;
    }
}

string data_path = "C:\\Users\\adria\\PycharmProjects\\BarcCluster\\data\\";
string input_file = "matrix_2000_both";

int main(int argc, const char** argv) {
    // set some bigger buffer so that fstream doesn't copy 1 char at a time
    const size_t bufsize = 256*1024;
    char buf[bufsize];

    ifstream f1(data_path+"scoring_scheme.txt");
    f1.rdbuf()->pubsetbuf(buf, bufsize);
    int num_buckets, bucket_size;
    vector<double> scores;
    f1 >> num_buckets >> bucket_size;
    double x;
    while (f1 >> x) scores.push_back(x);
    f1.close();

    ifstream f2(data_path+input_file+".txt");
    f2.rdbuf()->pubsetbuf(buf, bufsize);

    int N, num_samples, read_len;
    f2 >> N >> num_samples >> read_len;
    vector<string> names(N);
    vector< vector<double> > windows[2];
    for (int i = 0; i < N; i++) {
        cout << "loading read " << i << "\n";
        string name;
        f2 >> names[i];
        windows[0].push_back(load_read(f2, read_len));
        if (num_samples == 2) {
            windows[1].push_back(load_read(f2, read_len));
        }
    }
    f2.close();


    int N_threads = 1;
    if (argc > 0) N_threads = atoi(argv[1]);
    // Subdivide the matrix computation evenly across the threads.
    vector<pair<int, int> > indices;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            indices.push_back( {i, j} );
        }
    }

    vector< vector<double> > D(N, vector<double>(N));
    int chunk_size = indices.size()/N_threads;
    vector<thread> workers(N_threads);

    ofstream out(data_path+input_file+".out");
    out << N << endl;
    for (int i = 0; i < N; i++) out << names[i] << endl;

    for (int t = 0; t < num_samples; t++) {
        vector< vector< vector<Sentinel> > > frontiers(N_threads, vector< vector<Sentinel> > (N));
        for (int i = 0; i < N_threads - 1; i++) {
            workers[i] = thread(AlignmentThread, ref(D), ref(indices), ref(windows[t]), ref(frontiers[i]),
                                ref(scores), bucket_size, i * chunk_size, (i + 1) * chunk_size);
        }
        workers[N_threads - 1] = thread(AlignmentThread, ref(D), ref(indices), ref(windows[t]),
                                        ref(frontiers[N_threads - 1]),
                                        ref(scores), bucket_size, (N_threads - 1) * chunk_size, indices.size());

        for (int i = 0; i < N_threads; i++) workers[i].join();

        for (int i = 1; i < frontiers.size(); i++) {
            for (int j = 0; j < frontiers[i].size(); j++) {
                for (int k = 0; k < frontiers[i][j].size(); k++) {
                    frontiers[0][j].push_back(frontiers[i][j][k]);
                }
            }
        }
        WriteMatrix(out, frontiers[0], D);
    }
    out.close();
}
