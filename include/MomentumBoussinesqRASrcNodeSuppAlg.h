// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MomentumBoussinesqRASrcNodeSuppAlg_h
#define MomentumBoussinesqRASrcNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra {
namespace nalu {

class Realm;

class MomentumBoussinesqRASrcNodeSuppAlg : public SupplementalAlgorithm
{
public:
  MomentumBoussinesqRASrcNodeSuppAlg(Realm& realm);

  virtual ~MomentumBoussinesqRASrcNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(double* lhs, double* rhs, stk::mesh::Entity node);

  double compute_alpha(double delta_t);
  double update_average(double avg, double newVal);

  ScalarFieldType* temperature_;
  ScalarFieldType* raTemperature_;
  std::string raName_;
  ScalarFieldType* dualNodalVolume_;
  double rhoRef_;
  double beta_;
  int nDim_;

  std::vector<double> gravity_;
};

} // namespace nalu
} // namespace sierra

#endif
