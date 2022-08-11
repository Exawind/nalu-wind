// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef BoussinesqNonIsoMomentumSrcNodeSuppAlg_h
#define BoussinesqNonIsoMomentumSrcNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra {
namespace nalu {

class Realm;

class BoussinesqNonIsoMomentumSrcNodeSuppAlg : public SupplementalAlgorithm
{
public:
  BoussinesqNonIsoMomentumSrcNodeSuppAlg(Realm& realm);

  virtual ~BoussinesqNonIsoMomentumSrcNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(double* lhs, double* rhs, stk::mesh::Entity node);

  VectorFieldType* coordinates_;
  ScalarFieldType* dualNodalVolume_;

  const double visc_;
  const double Cp;

  double beta;
  double rhoRef;
  double TRef;
  std::vector<double> gravity;
};

} // namespace nalu
} // namespace sierra

#endif
