/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
