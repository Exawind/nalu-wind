// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef COMPUTEMETRICTENSORELEMALGORITHM_H
#define COMPUTEMETRICTENSORELEMALGORITHM_H

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class ComputeMetricTensorElemAlgorithm : public Algorithm {
public:
  ComputeMetricTensorElemAlgorithm(Realm &realm, stk::mesh::Part *part);
  virtual ~ComputeMetricTensorElemAlgorithm() {}

  virtual void execute();

  VectorFieldType *coordinates_{nullptr};
  GenericFieldType *Mij_{nullptr};

  std::vector<double> ws_coordinates;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_det_j;
  std::vector<double> ws_Mij;
};

} // namespace nalu
} // namespace sierra

#endif
