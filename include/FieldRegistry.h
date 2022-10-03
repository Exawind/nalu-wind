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
  static FieldDefTypes query(int numStates, std::string name)
  {
    static FieldRegistry instance;
    const std::map<std::string, FieldDefTypes>* db;

    switch (numStates) {
    case 2: {
      db = &(instance.database_2_state_);
      break;
    }
    case 3: {
      db = &(instance.database_3_state_);
      break;
    }
    default:
      throw std::runtime_error("Unsupported number of reference states");
    }

    auto fieldDefIter = db->find(name);

    if (fieldDefIter == db->end()) {
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
  // Inorder to accomodate and embed the state information we ended up creating
  // two separate databases that are templated on the number of states required
  // by the time integration scheme
  // This was done to preserve the singelton/refernce lookup only behavior of
  // the FieldRegistry
  const std::map<std::string, FieldDefTypes>& database_2_state_;
  const std::map<std::string, FieldDefTypes>& database_3_state_;
};

} // namespace nalu
} // namespace sierra

#endif /* FIELDREGISTRY_H_ */
