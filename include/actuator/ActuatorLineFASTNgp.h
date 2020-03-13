// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORLINEFASTNGP_H_
#define ACTUATORLINEFASTNGP_H_

#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulkFAST.h>
#include <actuator/ActuatorFunctorsFAST.h>
#include <actuator/UtilitiesActuator.h>

namespace sierra
{
namespace nalu
{

struct ActuatorLineFastNGP{

  ActuatorLineFastNGP(const ActuatorMetaFAST& actMeta,
    ActuatorBulkFAST& actBulk,
    stk::mesh::BulkData& stkBulk);

  void operator()();

  void update();

  const ActuatorMetaFAST& actMeta_;
  ActuatorBulkFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
};

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORLINEFASTNGP_H_ */
