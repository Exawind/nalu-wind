// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeGeometryInteriorAlgorithm_h
#define ComputeGeometryInteriorAlgorithm_h

#include<Algorithm.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class ComputeGeometryInteriorAlgorithm : public Algorithm
{
public:

  ComputeGeometryInteriorAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  ~ComputeGeometryInteriorAlgorithm();
  
  void execute();

  const bool assembleEdgeAreaVec_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
