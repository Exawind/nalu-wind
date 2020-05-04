// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


/** @file ActuatorLineSimple.h
 *  @brief A class to couple Nalu with OpenFAST for actuator line simulations of
 * wind turbines
 *
 */

#ifndef ActuatorLineSimple_h
#define ActuatorLineSimple_h

#include <stk_util/parallel/ParallelVectorConcat.hpp>
#include "ActuatorSimple.h"

namespace sierra {
namespace nalu {

class Realm;

class ActuatorLineSimple : public ActuatorSimple
{
public:
  ActuatorLineSimple(Realm& realm, const YAML::Node& node);
  ~ActuatorLineSimple() = default;

  std::string get_class_name() override;

  void update_class_specific() override;

  void execute_class_specific(
    const int nDim,
    const stk::mesh::FieldBase* coordinates,
    stk::mesh::FieldBase* actuator_source,
    const stk::mesh::FieldBase* dual_nodal_volume) override;

  void create_point_info_map_class_specific() override;

  void calculate_alpha(
    Coordinates ws, 
    Coordinates zeroalphadir, 
    Coordinates spandir,
    Coordinates chordnormaldir, 
    double twist,
    double &alpha,
    Coordinates &ws2D);

  void calculate_cl_cd(
    double alpha,
    std::vector<double> aoatable,
    std::vector<double> cltable,
    std::vector<double> cdtable,
    double &cl,
    double &cd);

};

} // namespace nalu
} // namespace sierra

#endif
