// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssembleNodalGradUNonConformalAlgorithm_h
#define AssembleNodalGradUNonConformalAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

class Realm;

class AssembleNodalGradUNonConformalAlgorithm : public Algorithm
{
public:
  AssembleNodalGradUNonConformalAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    VectorFieldType* vectorQ,
    TensorFieldType* dqdx);

  ~AssembleNodalGradUNonConformalAlgorithm();

  void execute();

  VectorFieldType* vectorQ_;
  TensorFieldType* dqdx_;

  ScalarFieldType* dualNodalVolume_;
  GenericFieldType* exposedAreaVec_;

  std::vector<const stk::mesh::FieldBase*> ghostFieldVec_;
};

} // namespace nalu
} // namespace sierra

#endif
