// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORBULKFAST_H_
#define ACTUATORBULKFAST_H_

#include <actuator/ActuatorBulk.h>
#include "OpenFAST.h"

namespace sierra{
namespace nalu{

struct ActuatorMetaFAST : public ActuatorMeta{
  ActuatorMetaFAST(const ActuatorMeta& actMeta);

  // HOST ONLY
  fast::fastInputs fastInputs_;
  std::vector<std::string> turbineNames_;
  std::vector<std::string> turbineOutputFileNames_;
  bool filterLiftLineCorrection_;
  // TODO(psakiev) not certain these need to be dual views
  ActVectorDblDv epsilon_;
  ActVectorDblDv epsilonChord_;
  ActVectorDblDv epsilonTower_;

};

struct ActuatorBulkFAST : public ActuatorBulk
{};

}
}

#endif /* ACTUATORBULKFAST_H_ */
