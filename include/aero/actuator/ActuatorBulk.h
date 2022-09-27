// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORBULK_H_
#define ACTUATORBULK_H_

#include <aero/actuator/ActuatorTypes.h>
#include <aero/actuator/ActuatorSearch.h>
#include <Enums.h>
#include <vector>

namespace stk {
namespace mesh {
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

struct ActuatorInfoNGP;

/*! \brief Meta data for working with actuator fields
 * This is an example of meta data that will be used to construct an actuator
 * object and the resulting bulk data. This object lives on host but views can
 * be accessed on host or device. Specialization for different models is
 * intended via inheritance.
 *
 */

struct ActuatorMeta
{
  ActuatorMeta(
    int numTurbines, ActuatorType actType = ActuatorType::ActLinePointDrag);
  virtual ~ActuatorMeta() {}
  void add_turbine(const ActuatorInfoNGP& info);
  const int numberOfActuators_;
  const ActuatorType actuatorType_;
  int numPointsTotal_;
  bool isotropicGaussian_;
  std::vector<std::string> searchTargetNames_;
  stk::search::SearchMethod searchMethod_;
  ActScalarIntDv numPointsTurbine_;
  bool useFLLC_ = false;
  ActVectorDblDv epsilonChord_;
  ActVectorDblDv epsilon_;
  ActFixScalarBool entityFLLC_;
  ActScalarIntDv numNearestPointsFllcInt_;
};

/*! \brief Where field data is stored and accessed for actuators
 * This object lives on host but the views can be on host, device or both
 *
 * The object as a whole will be created and live on host, and specialization is
 * intended through inheritance.
 */
struct ActuatorBulk
{
  ActuatorBulk(const ActuatorMeta& actMeta);
  virtual ~ActuatorBulk() {}

  void stk_search_act_pnts(
    const ActuatorMeta& actMeta, stk::mesh::BulkData& stkBulk);
  void zero_source_terms(stk::mesh::BulkData& stkBulk);
  void parallel_sum_source_term(stk::mesh::BulkData& stkBulk);
  void compute_offsets(const ActuatorMeta& actMeta);
  Kokkos::RangePolicy<ActuatorFixedExecutionSpace>
  local_range_policy(const ActuatorMeta& actMeta);

  // HOST AND DEVICE DATA (DualViews)
  ActScalarIntDv turbIdOffset_;
  ActVectorDblDv pointCentroid_;
  ActVectorDblDv velocity_;
  ActVectorDblDv actuatorForce_;
  ActVectorDblDv epsilon_;
  ActScalarDblDv searchRadius_;
  ActScalarU64Dv coarseSearchPointIds_;
  ActScalarU64Dv coarseSearchElemIds_;

  // Filtered lifting line correction fields
  ActVectorDblDv relativeVelocity_;
  ActScalarDblDv relativeVelocityMagnitude_;
  ActVectorDblDv liftForceDistribution_;
  ActVectorDblDv deltaLiftForceDistribution_;
  ActVectorDblDv epsilonOpt_;
  ActVectorDblDv fllc_;

  // HOST ONLY DATA
  ActFixVectorDbl localCoords_;
  ActFixScalarBool pointIsLocal_;
  ActFixScalarInt localParallelRedundancy_;
  ActFixElemIds elemContainingPoint_;

  const int localTurbineId_;
};

} // namespace nalu
} // namespace sierra
#endif
