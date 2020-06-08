// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORPARSINGFAST_H_
#define ACTUATORPARSINGFAST_H_

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

struct ActuatorMeta;
struct ActuatorMetaFAST;

ActuatorMetaFAST
actuator_FAST_parse(const YAML::Node& y_node, const ActuatorMeta& actMeta);

} // namespace nalu
} // namespace sierra

#endif /* ACTUATORPARSINGFAST_H_ */
