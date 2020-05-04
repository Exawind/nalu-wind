// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORBULKSIMPLE_H_
#define ACTUATORBULKSIMPLE_H_

#include <actuator/ActuatorBulk.h>

namespace sierra {
namespace nalu {

struct ActuatorMetaSimple : public ActuatorMeta
{
  ActuatorMetaSimple(const ActuatorMeta& actMeta);

  // HOST ONLY
  bool filterLiftLineCorrection_;
  bool isotropicGaussian_;

  // TODO(SAKIEVICH) not certain all these need to be dual views
  //int maxNumPntsPerBlade_;
  ActVectorDblDv epsilon_;
  ActVectorDblDv epsilonChord_;

  // Stuff for the simple blade
  bool            debug_output_;
  std::size_t     n_simpleblades_;
  ActScalarIntDv  num_force_pts_blade_;
  ActVectorDblDv  p1_;  // Start of blade
  ActVectorDblDv  p2_;  // End of blade
  ActVectorDblDv  p1zeroalphadir_;         // Directon of zero alpha at p1
  ActVectorDblDv  chordnormaldir_;         // Direction normal to chord
  ActVectorDblDv  spandir_;                // Direction in the span
    
  // Kokkos quantities
  // for the blade definitions
  std::size_t      max_num_force_pts_blade_;
  Act2DArrayDblDv  chord_tableDv_;
  Act2DArrayDblDv  twist_tableDv_;
  Act2DArrayDblDv  elem_areaDv_;
  // for the polars
  std::size_t      max_polartable_size_;
  ActScalarIntDv   polartable_size_;
  Act2DArrayDblDv  aoa_polartableDv_;
  Act2DArrayDblDv  cl_polartableDv_;
  Act2DArrayDblDv  cd_polartableDv_;

};

struct ActuatorBulkSimple : public ActuatorBulk
{
  ActuatorBulkSimple(const ActuatorMetaSimple& actMeta, double naluTimeStep);

  Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> local_range_policy();

  void init_epsilon(const ActuatorMetaSimple& actMeta);
  void init_points(const ActuatorMetaSimple& actMeta);
  void init_orientation(const ActuatorMetaSimple& actMeta);
  virtual void zero_open_fast_views();

  virtual ~ActuatorBulkSimple();

  ActFixVectorDbl turbineThrust_;

  ActVectorDblDv epsilonOpt_;
  ActTensorDblDv orientationTensor_;

  // Stuff for simple blade
  ActScalarIntDv  num_force_pts_blade_;
  ActScalarIntDv  assignedProc_;
  const int       num_blades_;
  const bool      debug_output_;

  // TODO(SAKIEVICH) this kill lambdas that are pass by value (KOKKOS_LAMBDA)
  // may need to rethink functor/bulk design.  Perhaps have an internal object
  // in bulk for gpu data and pass that into the actuatorFunctors.
  const int localTurbineId_;
  ActDualViewHelper<ActuatorMemSpace> dvHelper_;
};


} // namespace nalu
} // namespace sierra

#endif /* ACTUATORBULKSIMPLE_H_ */
