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
  WallDistEquationSystem(EquationSystems&, std::string = "");

  virtual ~WallDistEquationSystem();

  virtual void load(const YAML::Node&);

  void initial_work();

  void register_nodal_fields(stk::mesh::Part*);

  void register_edge_fields(stk::mesh::Part*);

  void register_element_fields(stk::mesh::Part*, const stk::topology&);

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

  static std::string min_wall_distance_name(std::string wallName)
  {
    return "minimum_distance_to_" + wallName;
  }
  static std::string wall_distance_phi_name(std::string wallName)
  {
    return wallName + "_distance_phi";
  }

  static std::string wall_distance_phi_bc_name(std::string wallName)
  {
    return wallName + "_distance_phi_bc";
  }

  static std::string dphidx_name(std::string wallName)
  {
    return "grad_" + wall_distance_phi_name(wallName);
  }

  void register_nodal_grad_algorithm_on_part(stk::mesh::Part* part);
  void register_disting_surface(stk::mesh::Part* part, bool = false);

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

  std::string wallName_{""};
};

class ComputeDistanceToSurface
{
public:
  ComputeDistanceToSurface(
    Realm& realm, 
    std::string surface_name,
    stk::mesh::PartVector interior,
    stk::mesh::PartVector bc)
    : meta_(*realm.metaData_),
      eqsys(realm.equationSystems_, surface_name),
      surface_name_(surface_name),
      interior_(interior),
      bc_(bc)
  {
  }

  void register_fields();
  void create_algorithms();
  const ScalarFieldType& compute();
private:
  stk::mesh::MetaData& meta_;
  WallDistEquationSystem eqsys;
  const std::string surface_name_;

  stk::mesh::PartVector interior_;
  stk::mesh::PartVector bc_;
};

} // namespace nalu
} // namespace sierra

#endif /* WALLDISTEQUATIONSYSTEM_H */
