//
// Created by adria on 3/11/2020.
//


#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>
#include "dtw.h"
using namespace std;


void AlignmentThread(vector<vector<double> > &D,
                     const vector<pair<int, int> > &indices,
                     const vector< vector<double> > &reads1,
                     const vector< vector<double> > &reads2,
                     const vector<double> &scores,
                     const int bucket_size,
		     const double delta,
	       	     const double score_coef,
		     const double path_coef,
                     const int start, const int end) {
    for (int idx = start; idx < end; idx++) {
        int i = indices[idx].first, j = indices[idx].second;
        //cout << "computing " << i << " " << j; 
        auto ans = LikelihoodAlignment(reads1[i], reads2[j], scores, bucket_size, delta, score_coef, path_coef);

        D[i][j] = get<0>(ans);
	//cout << " score: " << D[i][j] << endl;
    }
}

auto LoadScoringScheme(string filename) {
    cout << "Loading scoring scheme ...";
    const size_t bufsize = 256*1024;
    char buf[bufsize];
    ifstream f(filename);
    f.rdbuf()->pubsetbuf(buf, bufsize);
    int num_buckets, bucket_size;
    vector<double> scores;
    f >> num_buckets >> bucket_size;
    double x;
    while (f >> x) scores.push_back(x);
    f.close();
    cout << " finished" << endl;
    return scores;
}

vector<vector<vector<double> > > ComputeMatrix(const vector<vector< vector<double> > > &windows,
					       const string score_filename,
					       const double score_coef,
					       const double path_coef,
					       const int bucket_size,
					       const double delta,
					       const int N_threads = 1) {
    auto scoring_scheme = LoadScoringScheme(score_filename);
    /*
    cout << scoring_scheme.size() << endl;
    for (int i = 0; i < scoring_scheme.size(); i++) {
	    cout << i << "  " << scoring_scheme[i] << " ";
    }
    cout << endl;
    */

    int num_samples = windows.size();
    int N = windows[0].size();
    // Subdivide the matrix computation evenly across the threads.
    vector<pair<int, int> > indices;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            indices.push_back( {i, j} );
        }
    }

    int chunk_size = indices.size()/N_threads;
    vector<vector< vector<double> > > D(2, vector<vector<double > > (N, vector<double>(N)));
    vector<thread> workers(N_threads);

    for (int t = 0; t < num_samples; t++) {
        for (int i = 0; i < N_threads - 1; i++) {
	    //cout << "Spawning worker " << i << endl;
            workers[i] = thread(AlignmentThread, ref(D[t]), ref(indices), ref(windows[t]), ref(windows[t]),
                                ref(scoring_scheme), bucket_size, delta, score_coef, path_coef, i * chunk_size, (i + 1) * chunk_size);
        }
        workers[N_threads - 1] = thread(AlignmentThread, ref(D[t]), ref(indices), ref(windows[t]), ref(windows[t]),
                                        ref(scoring_scheme), bucket_size, delta, score_coef, path_coef, (N_threads - 1) * chunk_size, indices.size());

        for (int i = 0; i < N_threads; i++) workers[i].join();
    }
    for (int t = 0; t < num_samples; t++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < i; j++) {
                D[t][i][j] = D[t][j][i];
            }
        }
    }
    return D;
}

vector<vector<vector<double> > > AlignToRepresentatives(const vector<vector<vector<double> > > &representatives,
                                                        const vector<vector<vector<double> > > &windows,
						        const string score_filename,
							const double score_coef,
						        const double path_coef,
						        const int bucket_size,
							const double delta,
						        const int N_threads = 1) {
    auto scoring_scheme = LoadScoringScheme(score_filename);

    int num_samples = windows.size();
    int num_representatives = representatives[0].size();
    int num_windows = windows[0].size();
    // Subdivide the matrix computation evenly across the threads.
    vector<pair<int, int> > indices;
    for (int i = 0; i < num_windows; i++) {
        for (int j = 0; j < num_representatives; j++) {
            indices.push_back( {i, j} );
        }
    }

    int chunk_size = indices.size()/N_threads;
    vector<vector< vector<double> > > D(2, vector<vector<double > > (num_windows, vector<double>(num_representatives)));
    vector<thread> workers(N_threads);

    for (int t = 0; t < num_samples; t++) {
        for (int i = 0; i < N_threads - 1; i++) {
            workers[i] = thread(AlignmentThread, ref(D[t]), ref(indices), ref(windows[t]), ref(representatives[t]),
                                ref(scoring_scheme), bucket_size, delta, score_coef, path_coef, i * chunk_size, (i + 1) * chunk_size);
        }
        workers[N_threads - 1] = thread(AlignmentThread, ref(D[t]), ref(indices), ref(windows[t]), ref(representatives[t]),
                                        ref(scoring_scheme), bucket_size, delta, score_coef, path_coef, (N_threads - 1) * chunk_size, indices.size());

        for (int i = 0; i < N_threads; i++) workers[i].join();
    }
    return D;
}
