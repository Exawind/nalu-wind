// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeWallFrictionVelocityAlgorithm_h
#define ComputeWallFrictionVelocityAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class ComputeWallFrictionVelocityAlgorithm : public Algorithm
{
public:

  ComputeWallFrictionVelocityAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    const bool &useShifted);
  virtual ~ComputeWallFrictionVelocityAlgorithm();

  void execute();

  void zero_nodal_fields();

  void compute_utau(
      const double &up, const double &yp,
      const double &density, const double &viscosity,
      double &utau);
  
  void normalize_nodal_fields();

  const bool useShifted_;
  const double yplusCrit_;
  const double elog_;
  const double kappa_;
  const int maxIteration_;
  const double tolerance_;

  VectorFieldType *velocity_;
  VectorFieldType *bcVelocity_;
  VectorFieldType *coordinates_;
  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  GenericFieldType *exposedAreaVec_;
  GenericFieldType *wallFrictionVelocityBip_;
  GenericFieldType *wallNormalDistanceBip_;
  ScalarFieldType *assembledWallNormalDistance_;
  ScalarFieldType *assembledWallArea_;
};

} // namespace nalu
} // namespace Sierra

#endif
