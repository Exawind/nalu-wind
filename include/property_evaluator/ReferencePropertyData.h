// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ReferencePropertyData_h
#define ReferencePropertyData_h

#include <Enums.h>

#include <vector>

namespace sierra {
namespace nalu {

class ReferencePropertyData
{
public:
  ReferencePropertyData();
  ~ReferencePropertyData();

  std::string speciesName_;
  double mw_;
  double massFraction_;
  double stoichiometry_;
  double primaryMassFraction_;
  double secondaryMassFraction_;
};

} // namespace nalu
} // namespace sierra

#endif
