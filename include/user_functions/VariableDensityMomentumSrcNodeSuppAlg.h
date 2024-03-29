// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef VariableDensityMomentumSrcNodeSuppAlg_h
#define VariableDensityMomentumSrcNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra {
namespace nalu {

class Realm;

class VariableDensityMomentumSrcNodeSuppAlg : public SupplementalAlgorithm
{
public:
  VariableDensityMomentumSrcNodeSuppAlg(Realm& realm);

  virtual ~VariableDensityMomentumSrcNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(double* lhs, double* rhs, stk::mesh::Entity node);

  VectorFieldType* coordinates_;
  ScalarFieldType* dualNodalVolume_;

  const int nDim_;
  const double unot_;
  const double vnot_;
  const double wnot_;
  const double pnot_;
  const double znot_;
  const double a_;
  const double amf_;
  const double visc_;
  const double rhoP_;
  const double rhoS_;
  const double pi_;
  const double twoThirds_;
  double rhoRef_;
  double gx_;
  double gy_;
  double gz_;

  // space for source terms
  std::vector<double> srcXi_;
};

} // namespace nalu
} // namespace sierra

#endif
