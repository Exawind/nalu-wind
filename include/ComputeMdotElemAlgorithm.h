// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeMdotElemAlgorithm_h
#define ComputeMdotElemAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class ComputeMdotElemAlgorithm : public Algorithm
{
public:

  ComputeMdotElemAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    const bool assembleMdotToEdge);
  ~ComputeMdotElemAlgorithm();

  void execute();
  void assemble_edge_mdot();

  const bool meshMotion_;
  const bool assembleMdotToEdge_;

  // extract fields; nodal
  VectorFieldType *velocityRTM_;
  VectorFieldType *Gpdx_;
  VectorFieldType *coordinates_;
  ScalarFieldType *pressure_;
  ScalarFieldType *density_;
  GenericFieldType *massFlowRate_;
  ScalarFieldType *edgeMassFlowRate_;

  const bool shiftMdot_;
  const bool shiftPoisson_;
};

} // namespace nalu
} // namespace Sierra

#endif
