// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssembleNodalGradUAlgorithmDriver_h
#define AssembleNodalGradUAlgorithmDriver_h

#include <AlgorithmDriver.h>

namespace sierra {
namespace nalu {

class Realm;

class AssembleNodalGradUAlgorithmDriver : public AlgorithmDriver
{
public:
  AssembleNodalGradUAlgorithmDriver(Realm& realm, const std::string dudxName);
  virtual ~AssembleNodalGradUAlgorithmDriver() {}

  virtual void pre_work();
  virtual void post_work();

  const std::string dudxName_;
};

} // namespace nalu
} // namespace sierra

#endif
