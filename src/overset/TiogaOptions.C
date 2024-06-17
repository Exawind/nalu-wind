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

namespace tioga_nalu {

void
TiogaOptions::load(const YAML::Node& node)
{
  if (node["symmetry_direction"])
    symmetryDir_ = node["symmetry_direction"].as<int>();

  if (node["reduce_fringes"])
    reduceFringes_ = node["reduce_fringes"].as<bool>();

  if (node["num_fringe"]) {
    hasNumFringe_ = true;
    nFringe_ = node["num_fringe"].as<int>();
  }

  if (node["num_exclude"]) {
    hasMexclude_ = true;
    mExclude_ = node["num_exclude"].as<int>();
  }

  if (node["cell_resolution_multiplier"]) {
    cellResMult_ = node["cell_resolution_multiplier"].as<double>();
  }
  if (node["node_resolution_multiplier"]) {
    nodeResMult_ = node["node_resolution_multiplier"].as<double>();
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

} // namespace tioga_nalu
