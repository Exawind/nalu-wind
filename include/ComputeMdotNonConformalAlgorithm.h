// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ComputeMdotNonConformalAlgorithm_h
#define ComputeMdotNonConformalAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

class Realm;

class ComputeMdotNonConformalAlgorithm : public Algorithm
{
public:
  ComputeMdotNonConformalAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    ScalarFieldType* pressure,
    VectorFieldType* Gjp);
  ~ComputeMdotNonConformalAlgorithm();

  void execute();

  ScalarFieldType* pressure_;
  VectorFieldType* Gjp_;
  VectorFieldType* velocity_;
  VectorFieldType* meshVelocity_;
  VectorFieldType* coordinates_;
  ScalarFieldType* density_;
  GenericFieldType* exposedAreaVec_;
  GenericFieldType* ncMassFlowRate_;

  const bool meshMotion_;
  const bool useCurrentNormal_;
  const double includePstab_;
  double meshMotionFac_;

  std::vector<const stk::mesh::FieldBase*> ghostFieldVec_;
};

} // namespace nalu
} // namespace sierra

#endif
