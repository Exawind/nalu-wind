// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FIELDREGISTRY_H_
#define FIELDREGISTRY_H_

#include <stdexcept>
#include <string>
#include <map>
#include "FieldDefinitions.h"

namespace sierra {
namespace nalu {

/* A class that contains definitions for all the available fields that can be
 * registered in nalu-wind
 */
class FieldRegistry
{
public:
  static FieldDefTypes query(std::string name)
  {
    static FieldRegistry instance;

    auto fieldDefIter = instance.database_.find(name);

    if (fieldDefIter == instance.database_.end()) {
      const std::string message =
        "Attempting to access an undefined field: " + name;
      throw std::runtime_error(message);
    }
    return fieldDefIter->second;
  }

  FieldRegistry(const FieldRegistry&) = delete;
  void operator=(FieldRegistry const&) = delete;

private:
  FieldRegistry();
  // TODO right now just fixing to second order
  const std::map<std::string, FieldDefTypes>& database_;
};

} // namespace nalu
} // namespace sierra

#endif /* FIELDREGISTRY_H_ */
