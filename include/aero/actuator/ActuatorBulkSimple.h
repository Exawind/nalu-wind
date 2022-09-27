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

#include <aero/actuator/ActuatorBulk.h>

namespace sierra {
namespace nalu {

struct ActuatorMetaSimple : public ActuatorMeta
{
  ActuatorMetaSimple(const ActuatorMeta& actMeta);
  virtual ~ActuatorMetaSimple() {}

  // HOST ONLY
  bool isotropicGaussian_;

  // Stuff for the simple blade
  bool debug_output_;
  bool useSpreadActuatorForce;
  std::size_t n_simpleblades_;
  ActScalarIntDv num_force_pts_blade_;
  ActVectorDblDv p1_;             // Start of blade
  ActVectorDblDv p2_;             // End of blade
  ActScalarDblDv dR_;             // Spacing between points
  ActVectorDblDv p1ZeroAlphaDir_; // Directon of zero alpha at p1
  ActVectorDblDv chordNormalDir_; // Direction normal to chord
  ActVectorDblDv spanDir_;        // Direction in the span

  // Kokkos quantities
  // for the blade definitions
  std::size_t max_num_force_pts_blade_;
  Act2DArrayDblDv chord_tableDv_;
  Act2DArrayDblDv twistTableDv_;
  Act2DArrayDblDv elemAreaDv_;
  // for the polars
  std::size_t maxPolarTableSize_;
  ActScalarIntDv polarTableSize_;
  Act2DArrayDblDv aoaPolarTableDv_;
  Act2DArrayDblDv clPolarTableDv_;
  Act2DArrayDblDv cdPolarTableDv_;

  std::vector<std::string> output_filenames_;
  bool has_output_file_;
};

struct ActuatorBulkSimple : public ActuatorBulk
{
  ActuatorBulkSimple(const ActuatorMetaSimple& actMeta);
  virtual ~ActuatorBulkSimple() {}

  Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> local_range_policy();

  void init_epsilon(const ActuatorMetaSimple& actMeta);
  void init_points(const ActuatorMetaSimple& actMeta);
  void init_orientation(const ActuatorMetaSimple& actMeta);
  void add_output_headers(const ActuatorMetaSimple& actMeta);
  virtual void zero_actuator_views();

  ActScalarDblDv density_;
  ActScalarDblDv alpha_;

  ActFixVectorDbl turbineThrust_;

  ActTensorDblDv orientationTensor_;

  // Stuff for simple blade
  ActScalarIntDv num_force_pts_blade_;
  ActScalarIntDv assignedProc_;
  const int num_blades_;
  const bool debug_output_;

  ActDualViewHelper<ActuatorMemSpace> dvHelper_;
  std::vector<std::string> output_cache_;
};

} // namespace nalu
} // namespace sierra

#endif /* ACTUATORBULKSIMPLE_H_ */
