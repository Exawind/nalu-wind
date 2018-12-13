/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WALLDISTEQUATIONSYSTEM_H
#define WALLDISTEQUATIONSYSTEM_H

#include "EquationSystem.h"
#include "FieldTypeDef.h"

#include <memory>

namespace sierra {
namespace nalu {

class Realm;
class AssembleNodalGradAlgorithmDriver;
class EquationSystems;

class WallDistEquationSystem : public EquationSystem
{
public:
  WallDistEquationSystem(EquationSystems&);

  virtual ~WallDistEquationSystem();

  virtual void load(const YAML::Node&);

  void initial_work();

  void register_nodal_fields(stk::mesh::Part*);

  void register_edge_fields(stk::mesh::Part*);

  void register_element_fields(stk::mesh::Part*, const stk::topology&);

  void register_interior_algorithm(stk::mesh::Part*);

  void register_inflow_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const InflowBoundaryConditionData&);

  void register_open_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const OpenBoundaryConditionData&);

  void register_wall_bc(
    stk::mesh::Part*, const stk::topology&, const WallBoundaryConditionData&);

  void register_symmetry_bc(
    stk::mesh::Part*,
    const stk::topology&,
    const SymmetryBoundaryConditionData&);

  virtual void register_non_conformal_bc(
    stk::mesh::Part*,
    const stk::topology&);

  virtual void register_overset_bc();

  virtual void create_constraint_algorithm(stk::mesh::FieldBase*);

  void solve_and_update();

  void initialize();
  void reinitialize_linear_system();

  int pValue() { return pValue_; }

private:
  WallDistEquationSystem() = delete;
  WallDistEquationSystem(const WallDistEquationSystem&) = delete;

  void compute_wall_distance();

  VectorFieldType* coordinates_{nullptr};
  ScalarFieldType* wallDistPhi_{nullptr};
  VectorFieldType* dphidx_{nullptr};
  ScalarFieldType* wallDistance_{nullptr};
  ScalarFieldType* dualNodalVolume_{nullptr};
  VectorFieldType* edgeAreaVec_{nullptr};

  std::unique_ptr<AssembleNodalGradAlgorithmDriver> assembleNodalGradAlgDriver_;

  int pValue_{2};

  //! Frequency (in timesteps) at which wall distance is updated
  int updateFreq_{1};

  const bool managePNG_;

  bool isInit_{true};
};

}  // nalu
}  // sierra


#endif /* WALLDISTEQUATIONSYSTEM_H */
