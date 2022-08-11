// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <iomanip>
#include <sstream>

#include "ngp_algorithms/NgpAlgDriver.h"
#include "Realm.h"

namespace sierra {
namespace nalu {

NgpAlgDriver::NgpAlgDriver(Realm& realm) : realm_(realm) {}

std::string
NgpAlgDriver::unique_name(
  AlgorithmType algType, std::string entityType, std::string algName)
{
  std::stringstream ss;

  ss << "Alg" << std::setfill('0') << std::setw(4) << static_cast<int>(algType)
     << "_" << entityType << "_" << algName;

  return ss.str();
}

void
NgpAlgDriver::execute()
{
  pre_work();

  for (auto& kv : algMap_) {
    kv.second->execute();
  }

  post_work();
}

void
NgpAlgDriver::pre_work()
{
}

void
NgpAlgDriver::post_work()
{
}

} // namespace nalu
} // namespace sierra
