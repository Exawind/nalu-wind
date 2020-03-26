// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


/** @file ActuatorLineFAST.h
 *  @brief A class to couple Nalu with OpenFAST for actuator line simulations of
 * wind turbines
 *
 */

#ifndef ActuatorLineFAST_h
#define ActuatorLineFAST_h

#include <stk_util/parallel/ParallelVectorConcat.hpp>
#include "ActuatorFAST.h"

// OpenFAST C++ API
#include "OpenFAST.H"

namespace sierra {
namespace nalu {

class Realm;

class ActuatorLineFAST : public ActuatorFAST
{
public:
  ActuatorLineFAST(Realm& realm, const YAML::Node& node);
  ~ActuatorLineFAST() = default;

  std::string get_class_name() override;

  void update_class_specific() override;

  void execute_class_specific(
    const int nDim,
    const stk::mesh::FieldBase* coordinates,
    stk::mesh::FieldBase* actuator_source,
    const stk::mesh::FieldBase* dual_nodal_volume) override;

  void create_point_info_map_class_specific() override;
};

} // namespace nalu
} // namespace sierra

#endif
