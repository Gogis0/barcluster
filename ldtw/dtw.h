//
// Created by adria on 1/27/2020.
//

#ifndef LIKELIDTW_DTW_H
#define LIKELIDTW_DTW_H

#include <vector>
#include <tuple>

extern std::vector<std::pair<int, int> > AlignmentPath(std::vector<std::vector<double> > &alignment_matrix,
                                                       std::pair<int, int> argmax);


extern std::tuple<double, int, int, int, int> LikelihoodAlignment(const std::vector<double> &signal1,
                                                                  const std::vector<double> &signal2,
                                                                  const std::vector<double> &scoring_scheme,
                                                                  const int bucket_size,
                                                                  const double delta = 0.5,
								  const double score_coef=1,
								  const double path_coef=0);

#endif //LIKELIDTW_DTW_H
