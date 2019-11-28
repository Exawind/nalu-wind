// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeGeometryBoundaryAlgorithm_h
#define ComputeGeometryBoundaryAlgorithm_h

#include<Algorithm.h>

namespace sierra{
namespace nalu{

class Realm;

class ComputeGeometryBoundaryAlgorithm : public Algorithm
{
public:

  ComputeGeometryBoundaryAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~ComputeGeometryBoundaryAlgorithm() {}

  virtual void execute();
};

} // namespace nalu
} // namespace Sierra

#endif
