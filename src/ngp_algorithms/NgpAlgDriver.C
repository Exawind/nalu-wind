/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <iomanip>
#include <sstream>

#include "ngp_algorithms/NgpAlgDriver.h"
#include "Realm.h"

namespace sierra {
namespace nalu {

NgpAlgDriver::NgpAlgDriver(
  Realm& realm
): realm_(realm),
   nDim_(realm_.spatialDimension_)
{}

std::string
NgpAlgDriver::unique_name(
  AlgorithmType algType,
  std::string entityType,
  std::string algName)
{
  std::stringstream ss;

  ss << "Alg"
     << std::setfill('0') << std::setw(4)
     << static_cast<int>(algType)
     << "_" << entityType << "_"
     << algName;

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
{}

void
NgpAlgDriver::post_work()
{}

}  // nalu
}  // sierra
