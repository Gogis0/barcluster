//
// Created by adria on 3/11/2020.
//

#ifndef LIKELIDTW_COMPUTEMATRIX_H
#define LIKELIDTW_COMPUTEMATRIX_H
#include <vector>
#include <string>

std::vector<std::vector<std::vector<double> > > ComputeMatrix(
               const std::vector<std::vector< std::vector<double> > > &windows,
	       const std::string score_filename,
	       const int bucket_size,
	       const int N_threads = 1);


std::vector<std::vector<std::vector<double> > > AlignToRepresentatives(const std::vector< std::vector<double> > &representatives,
				                                        const std::vector<std::vector<std::vector<double> > > &windows,
									const std::string score_filename,
									const int bucket_size,
									const int N_threads = 1);

#endif //LIKELIDTW_COMPUTEMATRIX_H
