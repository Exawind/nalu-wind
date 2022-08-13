// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORBULKFAST_H_
#define ACTUATORBULKFAST_H_

#include <aero/actuator/ActuatorBulk.h>
#include "OpenFAST.H"

namespace sierra {
namespace nalu {

struct ActuatorMetaFAST : public ActuatorMeta
{
  ActuatorMetaFAST(const ActuatorMeta& actMeta);

  // HOST ONLY
  fast::fastInputs fastInputs_;
  std::vector<std::string> turbineNames_;
  std::vector<std::string> turbineOutputFileNames_;
  bool is_disk();
  int get_fast_index(
    fast::ActuatorNodeType type,
    int turbId,
    int index = 0,
    int bladeNum = 0) const;

  // TODO(SAKIEVICH) not certain all these need to be dual views
  int maxNumPntsPerBlade_;
  ActVectorDblDv epsilonTower_;
  ActVectorDblDv epsilonHub_;
  ActFixScalarBool useUniformAziSampling_;
  ActFixScalarInt nPointsSwept_;
  ActFixScalarInt nBlades_;
};

struct ActuatorBulkFAST : public ActuatorBulk
{
  ActuatorBulkFAST(const ActuatorMetaFAST& actMeta, double naluTimeStep);

  Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> local_range_policy();

  void interpolate_velocities_to_fast();
  void step_fast();
  bool fast_is_time_zero();
  void output_torque_info(stk::mesh::BulkData& stkBulk);
  void
  init_openfast(const ActuatorMetaFAST& actMeta, const double naluTimeStep);
  void init_epsilon(const ActuatorMetaFAST& actMeta);
  bool is_tstep_ratio_admissable(
    const double fastTimeStep, const double naluTimeStep);

  virtual ~ActuatorBulkFAST();

  ActFixVectorDbl turbineThrust_;
  ActFixVectorDbl turbineTorque_;
  ActFixVectorDbl hubLocations_;
  ActFixVectorDbl hubOrientation_;

  ActTensorDblDv orientationTensor_;

  fast::OpenFAST openFast_;
  const int tStepRatio_;
  ActDualViewHelper<ActuatorMemSpace> dvHelper_;
};

// helper functions to
// squash calls to std::cout from OpenFAST
inline void
squash_fast_output(std::function<void()> func)
{
  std::stringstream buffer;
  std::streambuf* sHoldCout = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
  func();
  std::cout.rdbuf(sHoldCout);
}

} // namespace nalu
} // namespace sierra

#endif /* ACTUATORBULKFAST_H_ */
