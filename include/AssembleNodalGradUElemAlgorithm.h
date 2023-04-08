// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssembleNodalGradUElemAlgorithm_h
#define AssembleNodalGradUElemAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
class AssembleNodalGradUElemAlgorithm : public Algorithm
{
public:
  AssembleNodalGradUElemAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    VectorFieldType* velocity,
    GenericFieldType* dudx,
    const bool useShifted = false);
  virtual ~AssembleNodalGradUElemAlgorithm() {}

  virtual void execute();

  VectorFieldType* vectorQ_;
  GenericFieldType* dqdx_;
  const bool useShifted_;
};

} // namespace nalu
} // namespace sierra

#endif
