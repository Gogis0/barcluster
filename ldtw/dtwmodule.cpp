//
// Created by adria on 3/11/2020.
//


#include <cstdio>
#include <vector>
#include "dtw.h"
#include "computematrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(ldtw, m) {
    m.doc() = "A C++ extension for fast LDTW computation";
    m.def("ComputeMatrix", &ComputeMatrix, "computes LDTW on a matrix",
        py::arg("windows"), py::arg("scoring_scheme"), py::arg("score_coef"), py::arg("path_coef"),
	py::arg("bucket_size"), py::arg("delta"), py::arg("N_threads"));
    m.def("AlignToRepresentatives", &AlignToRepresentatives, "alignes the reads to the chose representatives",
        py::arg("representatives"), py::arg("windows"), py::arg("scoring_scheme"), py::arg("score_coef"),
	py::arg("path_coef"), py::arg("bucket_size"), py::arg("delta"), py::arg("N_threads"));
    m.def("LikelihoodAlignment", &LikelihoodAlignment, "performs a single LDTW alignment",
        py::arg("signal1"), py::arg("signal2"), py::arg("scoring_scheme"), py::arg("bucket_size"),
	py::arg("delta"), py::arg("score_coef"), py::arg("path_coef"));
}
