// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleNodalGradAlgorithmDriver_h
#define AssembleNodalGradAlgorithmDriver_h

#include <AlgorithmDriver.h>
#include <string>

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodalGradAlgorithmDriver : public AlgorithmDriver
{
public:

  AssembleNodalGradAlgorithmDriver(
    Realm &realm,
    const std::string & scalarQName,
    const std::string & dqdxName);
  ~AssembleNodalGradAlgorithmDriver();

  void pre_work();
  void post_work();

  const std::string scalarQName_;
  const std::string dqdxName_;
  
};
  

} // namespace nalu
} // namespace Sierra

#endif
