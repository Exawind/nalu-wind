// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATOR_PARSING_H_
#define ACTUATOR_PARSING_H_

#include <NaluParsing.h>
#include <actuator/ActuatorBulk.h>

namespace sierra{
namespace nalu{

class ActuatorMeta;

/*! \brief Parse parameters to construct meta data for actuator's
 *  Parse parameters and construct meta data for actuators.
 *  Intent is to divorce object creation/memory allocation from parsing
 *  to facilitate device compatibility.
 *
 *  This also has the added benefit of increasing unittest-ability.
 */
ActuatorMeta actuator_parse(const YAML::Node& y_node, stk::mesh::BulkData& stkBulk);
//TODO(psakiev) define meta data structure
void actuator_line_FAST_parse(const YAML::Node& y_node, ActuatorMeta& actMeta);
void actuator_disk_FAST_parse(const YAML::Node& y_node, ActuatorMeta& actMeta);
}
}

#endif
