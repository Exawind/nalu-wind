// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MomentumBuoyancySrcNodeSuppAlg_h
#define MomentumBuoyancySrcNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra {
namespace nalu {

class Realm;

class MomentumBuoyancySrcNodeSuppAlg : public SupplementalAlgorithm
{
public:
  MomentumBuoyancySrcNodeSuppAlg(Realm& realm);

  virtual ~MomentumBuoyancySrcNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(double* lhs, double* rhs, stk::mesh::Entity node);

  ScalarFieldType* densityNp1_;
  ScalarFieldType* hydroDensityNp1_;
  ScalarFieldType* dualNodalVolume_;
  VectorFieldType* buoyancySource_;
  int nDim_;
  double rhoRef_;
  bool rhoRefIsHydroDens_;
  bool useBalancedSource_;

  std::vector<double> gravity_;
};

} // namespace nalu
} // namespace sierra

#endif
