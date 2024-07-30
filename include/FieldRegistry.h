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
  static FieldDefTypes query(int numDim, int numStates, std::string name)
  {
    static FieldRegistry instance;
    const std::map<std::string, FieldDefTypes>* db;

    switch (numDim) {
    case 2: {
      switch (numStates) {
      case 2: {
        db = &(instance.database_2D_2_state_);
        break;
      }
      case 3: {
        db = &(instance.database_2D_3_state_);
        break;
      }
      default:
        throw std::runtime_error("Unsupported number of reference states for field " + name + " with " + std::to_string(numStates) + " states and " + std::to_string(numDim) + " dims.");
      }
      break;
    }
    case 3: {
      switch (numStates) {
      case 2: {
        db = &(instance.database_3D_2_state_);
        break;
      }
      case 3: {
        db = &(instance.database_3D_3_state_);
        break;
      }
      default:
        throw std::runtime_error("Unsupported number of reference states for field " + name + " with " + std::to_string(numStates) + " states and " + std::to_string(numDim) + " dims.");
      }
      break;
    }
    default:
      throw std::runtime_error(
        "Only 2 and 3 spatial dimensions are supported. Dim Given was " +
        std::to_string(numDim));
    }

    auto fieldDefIter = db->find(name);

    if (fieldDefIter == db->end()) {
      std::string message = "Attempting to access an undefined field: '" +
                            name + "' with spatial dimension " +
                            std::to_string(numDim) + " and number of states " +
                            std::to_string(numStates);
      throw std::runtime_error(message);
    }
    return fieldDefIter->second;
  }

  FieldRegistry(const FieldRegistry&) = delete;
  void operator=(FieldRegistry const&) = delete;

private:
  FieldRegistry();
  // In order to accomodate and embed the state and dimenstion information we
  // ended up creating four separate databases that are templated on the number
  // of states required by the time integration scheme This was done to preserve
  // the singelton/refernce lookup only behavior of the FieldRegistry
  const std::map<std::string, FieldDefTypes>& database_2D_2_state_;
  const std::map<std::string, FieldDefTypes>& database_2D_3_state_;
  const std::map<std::string, FieldDefTypes>& database_3D_2_state_;
  const std::map<std::string, FieldDefTypes>& database_3D_3_state_;
};

} // namespace nalu
} // namespace sierra

#endif /* FIELDREGISTRY_H_ */
