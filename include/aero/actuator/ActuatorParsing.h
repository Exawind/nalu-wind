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

#include <aero/actuator/ActuatorBulk.h>
namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

/*! \brief Parse parameters to construct meta data for actuator's
 *  Parse parameters and construct meta data for actuators.
 *  Intent is to divorce object creation/memory allocation from parsing
 *  to facilitate device compatibility.
 *
 *  This also has the added benefit of increasing unittest-ability.
 */
ActuatorMeta actuator_parse(const YAML::Node& y_node);

/**
 * @brief Parsing logic for epsilon for the FLLC setup
 *
 * @param turbId
 * @param y_node
 * @param actMeta
 */
void
epsilon_parsing(int turbId, const YAML::Node& y_node, ActuatorMeta& actMeta);

} // namespace nalu
} // namespace sierra

#endif
