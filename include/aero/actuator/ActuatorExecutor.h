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

#include <aero/actuator/ActuatorFLLC.h>
#include <aero/actuator/ActuatorBulk.h>
namespace sierra {
namespace nalu {

/**
 * @brief Interface class for actuator execution model
 *
 * This class should be compatible with ActuatorBulk and ActuatorMeta,
 * and only immplementations/models that are compatible with these data types
 * should interface with this class directly
 */
class ActuatorExecutor
{
public:
  ActuatorExecutor(const ActuatorMeta& actMeta, ActuatorBulk& actBulk);
  ActuatorExecutor() = delete;
  virtual ~ActuatorExecutor(){};
  virtual void operator()() = 0;
  void compute_fllc();
  void apply_fllc(ActuatorBulk& actBulk);

private:
  FilteredLiftingLineCorrection fLiftLineCorr_;
};

} // namespace nalu
} // namespace sierra
#endif /* ACTUATOREXECUTOR_H_ */
