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

#include <string>
#include <map>
#include "FieldDefinitions.h"

namespace sierra {
namespace nalu {

class FieldRegistry
{
public:
  static FieldDefTypes query(std::string name)
  {
    static FieldRegistry instance;

    return instance.database_.at(name);
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
