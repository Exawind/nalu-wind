// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeMdotInflowAlgorithm_h
#define ComputeMdotInflowAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class ComputeMdotInflowAlgorithm : public Algorithm
{
public:

  ComputeMdotInflowAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    bool useShifted);
  ~ComputeMdotInflowAlgorithm();

  void execute();

  const bool useShifted_;

  VectorFieldType *velocityBC_;
  ScalarFieldType *densityBC_;
  GenericFieldType *exposedAreaVec_;
};

} // namespace nalu
} // namespace Sierra

#endif
