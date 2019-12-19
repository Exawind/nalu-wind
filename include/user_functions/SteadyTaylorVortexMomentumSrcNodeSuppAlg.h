// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef SteadyTaylorVortexMomentumSrcNodeSuppAlg_h
#define SteadyTaylorVortexMomentumSrcNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class SteadyTaylorVortexMomentumSrcNodeSuppAlg : public SupplementalAlgorithm
{
public:

  SteadyTaylorVortexMomentumSrcNodeSuppAlg(
    Realm &realm);

  virtual ~SteadyTaylorVortexMomentumSrcNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
  VectorFieldType *coordinates_;
  ScalarFieldType *dualNodalVolume_;

  const int nDim_;
  const double unot_;
  const double a_;
  const double visc_;
  const double pi_;

  std::vector<double> srcXi_;
};

} // namespace nalu
} // namespace Sierra

#endif
