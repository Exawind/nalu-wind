// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORINFO_H_
#define ACTUATORINFO_H_

#include <NaluParsing.h>

namespace sierra {
namespace nalu {

/*! \brief Data structure to stash turbine info during parsing
 * Data structure to stash turbine info during parsing. This
 * object is intended to be specialized via inheritance for
 * specific actuator types. This is intended to be a limited
 * life span object.
 */
struct ActuatorInfoNGP
{
  int processorId_{0};
  int turbineId_{0};
  int numPoints_{0};
  Coordinates epsilon_;
  std::string turbineName_;
};

} // namespace nalu
} // namespace sierra

#endif
