// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TurbKineticEnergyEquationSystem_h
#define TurbKineticEnergyEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsedTypes.h>

#include "ngp_algorithms/NodalGradAlgDriver.h"
#include "ngp_algorithms/TKEWallFuncAlgDriver.h"

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

class Realm;
class LinearSystem;
class EquationSystems;
class ProjectedNodalGradientEquationSystem;

class TurbKineticEnergyEquationSystem : public EquationSystem
{

public:
  TurbKineticEnergyEquationSystem(EquationSystems& equationSystems);

  virtual ~TurbKineticEnergyEquationSystem() = default;

  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);

  void register_interior_algorithm(stk::mesh::Part* part);

  void register_inflow_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const InflowBoundaryConditionData& inflowBCData);

  void register_open_bc(
    stk::mesh::Part* part,
    const stk::topology& partTopo,
    const OpenBoundaryConditionData& openBCData);

  void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData);

  virtual void register_symmetry_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const SymmetryBoundaryConditionData& symmetryBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part* part, const stk::topology& theTopo);

  virtual void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();

  void predict_state();

  void solve_and_update();
  void initial_work();

  void compute_effective_diff_flux_coeff();
  void compute_wall_model_parameters();
  void update_and_clip();

  void manage_projected_nodal_gradient(EquationSystems& eqSystems);
  void compute_projected_nodal_gradient();

  void post_external_data_transfer_work();
  static bool check_for_valid_turblence_model(TurbulenceModel turbModel);

  const bool managePNG_;

  ScalarFieldType* tke_;
  VectorFieldType* dkdx_;
  ScalarFieldType* kTmp_;
  ScalarFieldType* visc_;
  ScalarFieldType* tvisc_;
  ScalarFieldType* evisc_;

  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  std::unique_ptr<TKEWallFuncAlgDriver> wallFuncAlgDriver_;
  std::unique_ptr<Algorithm> effDiffFluxCoeffAlg_;
  const TurbulenceModel turbulenceModel_;

  ProjectedNodalGradientEquationSystem* projectedNodalGradEqs_;

  bool isInit_;
};

} // namespace nalu
} // namespace sierra

#endif
