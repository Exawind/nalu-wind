// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef WALLDISTEQUATIONSYSTEM_H
#define WALLDISTEQUATIONSYSTEM_H

#include "EquationSystem.h"
#include "FieldTypeDef.h"
#include "ngp_algorithms/NodalGradAlgDriver.h"

#include <memory>

namespace sierra {
namespace nalu {

class Realm;
class EquationSystems;

class WallDistEquationSystem : public EquationSystem
{
public:
  WallDistEquationSystem(EquationSystems&);

  virtual ~WallDistEquationSystem();

  virtual void load(const YAML::Node&);

  void initial_work();

  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);
  virtual void register_edge_fields(const stk::mesh::PartVector& part_vec);
  virtual void register_element_fields(
    const stk::mesh::PartVector& part_vec, const stk::topology& theTopo);

  void register_interior_algorithm(stk::mesh::Part*);

  void register_inflow_bc(
    stk::mesh::Part*, const stk::topology&, const InflowBoundaryConditionData&);

  void register_open_bc(
    stk::mesh::Part*, const stk::topology&, const OpenBoundaryConditionData&);

  void register_wall_bc(
    stk::mesh::Part*, const stk::topology&, const WallBoundaryConditionData&);

  void register_symmetry_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const SymmetryBoundaryConditionData&);

  virtual void
  register_non_conformal_bc(stk::mesh::Part*, const stk::topology&);

  virtual void register_overset_bc();

  virtual void create_constraint_algorithm(stk::mesh::FieldBase*);

  void solve_and_update();

  void initialize();
  void reinitialize_linear_system();

  int pValue() { return pValue_; }

  void compute_wall_distance();

private:
  WallDistEquationSystem() = delete;
  WallDistEquationSystem(const WallDistEquationSystem&) = delete;

  VectorFieldType* coordinates_{nullptr};
  ScalarFieldType* wallDistPhi_{nullptr};
  VectorFieldType* dphidx_{nullptr};
  ScalarFieldType* wallDistance_{nullptr};
  ScalarFieldType* dualNodalVolume_{nullptr};
  VectorFieldType* edgeAreaVec_{nullptr};

  ScalarNodalGradAlgDriver nodalGradAlgDriver_;

  int pValue_{2};

  //! Frequency (in timesteps) at which wall distance is updated
  int updateFreq_{1};

  const bool managePNG_;

  bool isInit_{true};

  //! User option to force recomputation of wall distance on restart
  bool forceInitOnRestart_{false};
};

} // namespace nalu
} // namespace sierra

#endif /* WALLDISTEQUATIONSYSTEM_H */
