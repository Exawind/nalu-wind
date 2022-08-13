// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef AEROCONTAINER_H_
#define AEROCONTAINER_H_

#ifdef NALU_USES_OPENFAST
#include "OpenFAST.h"
// TODO add a blank openfast data structure possibly
#endif
#include "aero/actuator/ActuatorModel.h"

namespace sierra {
namespace nalu {

/**
 * A container class for holding all the aerodynamic models (actuators,
 * fsi-turbines, etc)
 */
class AeroContainer
{
private:
#ifdef NALU_USES_OPENFAST
  // TODO this should be the only instance of openfast
  // all other instances need to be made into shared_ptrs that share from this
  // one
  std::shared_ptr<fast::OpenFAST> fast_;
#endif
  // TODO move calls to actuator model in realm to this container class
  std::unique_ptr<ActuatorModel> actuatorModel_;
  const int tStepRatio_;
};

} // namespace nalu
} // namespace sierra

#endif
