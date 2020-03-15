// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNIT_TESTS_ACTUATOR_UNITTESTACTUATORUTIL_H_
#define UNIT_TESTS_ACTUATOR_UNITTESTACTUATORUTIL_H_

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

namespace sierra{
namespace nalu{
namespace actuator_unit{

static std::vector<std::string> nrel5MWinputs ={
    "actuator:\n",
    "  t_start: 0\n",
    "  simStart: init\n",
    "  n_every_checkpoint: 1\n",
    "  dt_fast: 0.00625\n",
    "  t_max: 0.0625\n",
    "  dry_run: no\n",
    "  debug: yes\n",
    "  Turbine0:\n",
    "    turbine_name: turbinator\n",
    "    epsilon: [5.0, 5.0, 5.00]\n",
    "    turb_id: 0\n",
    "    fast_input_filename: reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst\n",
    "    restart_filename: blah\n",
    "    num_force_pts_blade: 10\n",
    "    num_force_pts_tower: 10\n",
    "    turbine_base_pos: [0,0,0]\n",
    "    air_density:  1.0\n",
    "    nacelle_area:  1.0\n",
    "    nacelle_cd:  1.0\n",
};

inline
YAML::Node
create_yaml_node(const std::vector<std::string>& testFile)
{
  std::string temp;
  for (auto&& line : testFile) {
    temp += line;
  }
  return YAML::Load(temp);
}

}
} //namespace nalu
} //namespace sierra

#endif // UNIT_TESTS_ACTUATOR_UNITTESTACTUATORUTIL_H_
