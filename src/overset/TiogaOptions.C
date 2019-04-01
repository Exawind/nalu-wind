/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "overset/TiogaOptions.h"

#include "tioga.h"

namespace tioga_nalu {

void TiogaOptions::load(const YAML::Node& node)
{
  if (node["symmetry_direction"])
    symmetryDir_ = node["symmetry_direction"].as<int>();

  if (node["set_resolutions"])
    setResolutions_ = node["set_resolutions"].as<bool>();

  if (node["reduce_fringes"])
    reduceFringes_ = node["reduce_fringes"].as<bool>();

  if (node["num_fringe"]) {
    hasNumFringe_ = true;
    nFringe_ = node["num_fringe"].as<int>();
  }

  if (node["num_exclude"]) {
    hasMexclude_ = true;
    mExclude_= node["num_exclude"].as<int>();
  }
}

void TiogaOptions::set_options(TIOGA::tioga& tg)
{
  tg.setSymmetry(symmetryDir_);

  if (hasMexclude_) tg.setMexclude(&mExclude_);

  if (hasNumFringe_) tg.setNfringe(&nFringe_);
}

}
