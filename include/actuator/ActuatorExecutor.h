// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATOREXECUTOR_H_
#define ACTUATOREXECUTOR_H_

namespace sierra{
namespace nalu{

/**
 * @brief Interface class for actuator execution model
 * 
 */
struct ActuatorExecutor{
  ActuatorExecutor() = default;
  virtual ~ActuatorExecutor(){};
  virtual void operator()()=0;
};

}
}
#endif /* ACTUATOREXECUTOR_H_ */
