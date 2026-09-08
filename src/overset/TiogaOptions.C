// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "overset/TiogaOptions.h"

#include "tioga.h"

namespace tioga_kynema_ugf {

void
TiogaOptions::load(const YAML::Node& node)
{
  const YAML::Node symmetryDirection = node["symmetry_direction"];
  if (symmetryDirection)
    symmetryDir_ = symmetryDirection.as<int>();

  const YAML::Node reduceFringes = node["reduce_fringes"];
  if (reduceFringes)
    reduceFringes_ = reduceFringes.as<bool>();

  const YAML::Node numFringe = node["num_fringe"];
  if (numFringe) {
    hasNumFringe_ = true;
    nFringe_ = numFringe.as<int>();
  }

  const YAML::Node numExclude = node["num_exclude"];
  if (numExclude) {
    hasMexclude_ = true;
    mExclude_ = numExclude.as<int>();
  }

  const YAML::Node cellResolutionMultiplier =
    node["cell_resolution_multiplier"];
  if (cellResolutionMultiplier) {
    cellResMult_ = cellResolutionMultiplier.as<double>();
  }

  const YAML::Node nodeResolutionMultiplier =
    node["node_resolution_multiplier"];
  if (nodeResolutionMultiplier) {
    nodeResMult_ = nodeResolutionMultiplier.as<double>();
  }
}

void
TiogaOptions::set_options(TIOGA::tioga& tg)
{
  tg.setSymmetry(symmetryDir_);

  if (hasMexclude_)
    tg.setMexclude(&mExclude_);

  if (hasNumFringe_)
    tg.setNfringe(&nFringe_);
}

} // namespace tioga_kynema_ugf
