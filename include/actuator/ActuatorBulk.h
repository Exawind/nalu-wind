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
#include <stk_search/SearchMethod.hpp>

namespace sierra{
namespace nalu{

class ActuatorInfo;

#ifdef ACTUATOR_ON_DEVICE
using ActuatorExecSpace = Kokkos::CudaSpace;
using ActuatorMemLayout = Kokkos::LayoutRight;
#else
using ActuatorExecSpace = Kokkos::HostSpace;
using ActuatorMemLayout = Kokkos::LayoutLeft;
#endif


using ActScalarInt = Kokkos::DualView<int*,    ActuatorMemLayout, ActuatorExecSpace>;
using ActScalarDbl = Kokkos::DualView<double*, ActuatorMemLayout, ActuatorExecSpace>;
using ActVectorDbl = Kokkos::DualView<double*, ActuatorMemLayout, ActuatorExecSpace>;


//TODO(psakiev) Allocate bulk fields based on parameters

/*! \brief Meta data for working with actuator fields
 * This is an example of meta data that will be used to construct an actuator object
 * and the resulting bulk data.
 * The meta data should be copyable.
 */

class ActuatorMeta{
public:
  ActuatorMeta(int numTurbines, stk::search::SearchMethod searchMethod=stk::search::KDTREE);
  void add_turbine(int turbineIndex, const ActuatorInfo& info);
  inline int num_actuators() const {return numberOfActuators_;}
  inline int total_num_points(int i) const {return numPointsTotal_(i);}
private:
  const int numberOfActuators_;
  stk::search::SearchMethod searchMethod_;
  ActScalarInt numPointsTotal_;
};

/// Where field data is stored and accessed for actuators
class ActuatorBulk{
public:
  ActuatorBulk(ActuatorMeta meta);
  inline int total_num_points(){ return totalNumPoints_;}
private:
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
