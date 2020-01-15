// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleWallHeatTransferAlgorithmDriver_h
#define AssembleWallHeatTransferAlgorithmDriver_h

#include <AlgorithmDriver.h>
#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleWallHeatTransferAlgorithmDriver : public AlgorithmDriver
{
public:

  AssembleWallHeatTransferAlgorithmDriver(
    Realm &realm);
  ~AssembleWallHeatTransferAlgorithmDriver();

  void pre_work();
  void post_work();

  ScalarFieldType *assembledWallArea_;
  ScalarFieldType *referenceTemperature_;
  ScalarFieldType *heatTransferCoefficient_;
  ScalarFieldType *normalHeatFlux_;
  ScalarFieldType *robinCouplingParameter_;
  ScalarFieldType *temperature_;

};
  

} // namespace nalu
} // namespace Sierra

#endif
