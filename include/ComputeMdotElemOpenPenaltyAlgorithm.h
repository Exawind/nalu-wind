// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeMdotElemOpenPenaltyAlgorithm_h
#define ComputeMdotElemOpenPenaltyAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;
class MdotAlgDriver;

class ComputeMdotElemOpenPenaltyAlgorithm : public Algorithm
{
public:

  ComputeMdotElemOpenPenaltyAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    MdotAlgDriver& mdotDriver);
  ~ComputeMdotElemOpenPenaltyAlgorithm();

  void execute();

  MdotAlgDriver& mdotDriver_;
  VectorFieldType *velocityRTM_;
  VectorFieldType *Gpdx_;
  VectorFieldType *coordinates_;
  ScalarFieldType *pressure_;
  ScalarFieldType *density_;
  GenericFieldType *exposedAreaVec_;
  GenericFieldType *openMassFlowRate_;
  ScalarFieldType *pressureBc_;

  const double interpTogether_;
  const double om_interpTogether_;
  const bool shiftMdot_;
  const bool shiftedGradOp_;
  const double stabFac_;
};

} // namespace nalu
} // namespace Sierra

#endif
