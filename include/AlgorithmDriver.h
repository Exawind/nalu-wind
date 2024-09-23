// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AlgorithmDriver_h
#define AlgorithmDriver_h

#include <Enums.h>

#include <map>

namespace sierra {
namespace nalu {

class Realm;
class Algorithm;

class AlgorithmDriver
{
public:
  AlgorithmDriver(Realm& realm);
  virtual ~AlgorithmDriver();

  virtual void pre_work() {};
  virtual void execute();
  virtual void post_work() {};

  Realm& realm_;
  std::map<AlgorithmType, Algorithm*> algMap_;
};

} // namespace nalu
} // namespace sierra

#endif
