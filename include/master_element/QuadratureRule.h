// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef QuadratureRule_h
#define QuadratureRule_h

#include <vector>

namespace sierra {
namespace nalu {

// <abscissae, weights>
std::pair<std::vector<double>, std::vector<double>>
gauss_legendre_rule(int order);

// <abscissae, weights>
std::pair<std::vector<double>, std::vector<double>> gauss_lobatto_legendre_rule(
  int order, double xleft = -1.0, double xright = +1.0);

// <abscissae, weights>
std::pair<std::vector<double>, std::vector<double>>
SGL_quadrature_rule(int order, const double* scsEndLocations);

// a vector with -1 added at the first entry and +1 added at the last entry
std::vector<double> pad_end_points(
  const std::vector<double>& x, double xleft = -1.0, double xright = +1.0);
std::vector<double> pad_end_points(
  int n, const double* x, double xleft = -1.0, double xright = +1.0);

} // namespace nalu
} // namespace sierra

#endif
