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

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace sierra{
namespace nalu{

class ActuatorInfoNGP;

#ifdef ACTUATOR_ON_DEVICE
using ActuatorMemSpace = Kokkos::CudaSpace;
using ActuatorMemLayout = Kokkos::LayoutRight;
#else
using ActuatorMemSpace = Kokkos::HostSpace;
using ActuatorMemLayout = Kokkos::LayoutLeft;
#endif


using ActScalarInt = Kokkos::DualView<int*,    ActuatorMemLayout, ActuatorMemSpace>;
using ActScalarDbl = Kokkos::DualView<double*, ActuatorMemLayout, ActuatorMemSpace>;
using ActVectorDbl = Kokkos::DualView<double**, ActuatorMemLayout, ActuatorMemSpace>;


//TODO(psakiev) Allocate bulk fields based on parameters

/*! \brief Meta data for working with actuator fields
 * This is an example of meta data that will be used to construct an actuator object
 * and the resulting bulk data.
 * The meta data should be copyable.
 */

class ActuatorMeta{
public:
  ActuatorMeta(int numTurbines);
  void add_turbine(int turbineIndex, const ActuatorInfoNGP& info);
  inline int num_actuators() const {return numberOfActuators_;}
  inline int total_num_points(int i) const {return numPointsTotal_.h_view(i);}
private:
  const int numberOfActuators_;
  ActScalarInt numPointsTotal_;
};

/// Where field data is stored and accessed for actuators
class ActuatorBulk{
public:
  ActuatorBulk(ActuatorMeta meta);
  const ActuatorMeta actuatorMeta_;
  const int totalNumPoints_;
  ActVectorDbl pointCentroid_;
  ActVectorDbl velocity_;
  ActVectorDbl actuatorForce_;
  ActVectorDbl epsilon_;
};

}
}
#endif
