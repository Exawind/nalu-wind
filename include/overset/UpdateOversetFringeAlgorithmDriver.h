// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef UPDATEOVERSETFRINGEALGORITHMDRIVER_H
#define UPDATEOVERSETFRINGEALGORITHMDRIVER_H

#include "AlgorithmDriver.h"
#include "overset/OversetFieldData.h"

#include <memory>
#include <vector>

namespace stk {
namespace mesh {
class FieldBase;
}
}

namespace sierra {
namespace nalu {

class Realm;

class UpdateOversetFringeAlgorithmDriver : public AlgorithmDriver
{
public:
  UpdateOversetFringeAlgorithmDriver(Realm& realm);

  virtual ~UpdateOversetFringeAlgorithmDriver();

  virtual void execute() override;

  void register_overset_field_update(stk::mesh::FieldBase*, int, int);

  std::vector<OversetFieldData> fields_;
};

}  // nalu
}  // sierra

#endif /* UPDATEOVERSETFRINGEALGORITHMDRIVER_H */
