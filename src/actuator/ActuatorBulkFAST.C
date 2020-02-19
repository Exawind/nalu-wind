// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBulkFAST.h>

namespace sierra{
namespace nalu{

ActuatorMetaFAST::ActuatorMetaFAST(const ActuatorMeta& actMeta):
  ActuatorMeta(actMeta),
  turbineNames_(numberOfActuators_),
  turbineOutputFileNames_(numberOfActuators_),
  filterLiftLineCorrection_(false),
  epsilon_("epsilonMeta", numberOfActuators_),
  epsilonChord_("epsilonChordMeta", numberOfActuators_),
  epsilonTower_("epsilonTowerMeta", numberOfActuators_)
{}

}
}
