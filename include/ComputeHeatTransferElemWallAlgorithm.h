// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeHeatTransferElemWallAlgorithm_h
#define ComputeHeatTransferElemWallAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class ComputeHeatTransferElemWallAlgorithm : public Algorithm
{
public:

  ComputeHeatTransferElemWallAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  ~ComputeHeatTransferElemWallAlgorithm();

  void execute();

  ScalarFieldType *temperature_;
  VectorFieldType *coordinates_;
  ScalarFieldType *density_;
  ScalarFieldType *thermalCond_;
  ScalarFieldType *specificHeat_;
  GenericFieldType *exposedAreaVec_;
  ScalarFieldType *assembledWallArea_;
  ScalarFieldType *referenceTemperature_;
  ScalarFieldType *heatTransferCoefficient_;
  ScalarFieldType *normalHeatFlux_;
  ScalarFieldType *robinCouplingParameter_;

  double compute_coupling_parameter(const double & kappa,
                                    const double & h,
                                    const double & chi);  
};

} // namespace nalu
} // namespace Sierra

#endif
