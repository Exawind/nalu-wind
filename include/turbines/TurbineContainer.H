// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef TURBINECONTAINER_H_
#define TURBINECONTAINER_H_

#ifdef NALU_USES_OPENFAST
#include "OpenFAST.h"
// TODO add a blank openfast data structure possibly
#endif

namespace sierra {
namespace nalu {

/* Class should contain parsing element
 * an instance of openfast
 * a vector of turbines
 *
 *
 *
 */
class TurbineContainer
{
private:
#ifdef NALU_USES_OPENFAST
  fast::OpenFAST fast_;
#endif
  const int tStepRatio_;
};

} // namespace nalu
} // namespace sierra

#endif
