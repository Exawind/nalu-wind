/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifdef NALU_USES_OPENFAST

#include <ActuatorLineFAST.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>
#include <nalu_make_unique.h>

// master elements
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// basic c++
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>

namespace sierra{
namespace nalu{

// constructor
ActuatorLineFAST::ActuatorLineFAST(
  Realm &realm,
  const YAML::Node &node)
  : ActuatorFAST(realm, node)
{}

ActuatorLineFAST::~ActuatorLineFAST()
{
  FAST.end(); // Call destructors in FAST_cInterface
}

void ActuatorLineFAST::load_class_specific( const YAML::Node& node){}

void ActuatorLineFAST::update_class_specific(){
  ActuatorFAST::update();
}

void ActuatorLineFAST::create_point_info_map_class_specific(){}

void ActuatorLineFAST::execute_class_specific(){}

std::string ActuatorLineFAST::get_class_name(){
  return "ActuatorLineFAST";
}

} // namespace nalu
} // namespace Sierra

#endif
