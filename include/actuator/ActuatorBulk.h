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
#include <stk_search/SearchMethod.hpp>
#include <vector>

namespace sierra{
namespace nalu{

class ActuatorInfoNGP;

/*! \brief Meta data for working with actuator fields
 * This is an example of meta data that will be used to construct an actuator object
 * and the resulting bulk data. This object lives on host but views can be
 * accessed on host or device. Specialization for different models is intended
 * via inheritance.
 *
 * The meta data should be copyable.
 */

struct ActuatorMeta{
  ActuatorMeta(int numTurbines);
  void add_turbine(const ActuatorInfoNGP& info);
  //TODO(psakiev) do we want/need private members and accessor functions?
  inline int num_points_turbine(int i) const {return numPointsTurbine_.h_view(i);}
  const int numberOfActuators_;
  int numPointsTotal_;
  std::vector<std::string> searchTargetNames_;
  stk::search::SearchMethod searchMethod_;
  ActScalarIntDv numPointsTurbine_;
};

/*! \brief Where field data is stored and accessed for actuators
 * This object lives on host but the views can be on host, device or both
 * for now they are dual views but these can be specialized as implementation
 * desires.
 *
 * The object as a whole will be created and live on host, and specialization is
 * intended through inheritance.
 */
struct ActuatorBulk{
  ActuatorBulk(ActuatorMeta actMeta);
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
