// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleCourantReynoldsElemAlgorithm_h
#define AssembleCourantReynoldsElemAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleCourantReynoldsElemAlgorithm : public Algorithm
{
public:

  AssembleCourantReynoldsElemAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~AssembleCourantReynoldsElemAlgorithm() {}

  virtual void execute();
  
  const bool meshMotion_;

  VectorFieldType *velocityRTM_;
  VectorFieldType *coordinates_;
  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  GenericFieldType *elemReynolds_;
  GenericFieldType *elemCourant_;
};

} // namespace nalu
} // namespace Sierra

#endif
