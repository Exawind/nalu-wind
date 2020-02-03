// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFIELDBULK_H_
#define ACTUATORFIELDBULK_H_

#include <actuator/ActuatorTypes.h>

namespace sierra{
namespace nalu{

class ActuatorInfoNGP;

/*! \brief Meta data for working with actuator fields
 * This is an example of meta data that will be used to construct an actuator object
 * and the resulting bulk data. This object lives on host.
 *
 * The meta data should be copyable.
 */

class ActuatorMeta{
public:
  ActuatorMeta(int numTurbines);
  void add_turbine(int turbineIndex, const ActuatorInfoNGP& info);
  inline int num_actuators() const {return numberOfActuators_;}
  inline int num_points_total() const {return numPointsTotal_;}
  inline int num_points_turbine(int i) const {return numPointsTurbine_.h_view(i);}
private:
  const int numberOfActuators_;
  int numPointsTotal_;
  ActScalarIntDv numPointsTurbine_;
};

/*! \brief Where field data is stored and accessed for actuators
 * This object lives on host but the views can be on host, device or both
 */
struct ActuatorBulk{
  ActuatorBulk(ActuatorMeta meta);
  const ActuatorMeta actuatorMeta_;
  const int totalNumPoints_;
  ActVectorDblDv pointCentroid_;
  ActVectorDblDv velocity_;
  ActVectorDblDv actuatorForce_;
  ActVectorDblDv epsilon_;
};

}
}
#endif
