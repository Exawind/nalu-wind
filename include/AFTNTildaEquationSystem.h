// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#ifndef AFTNTildaEquationSystem_h
#define AFTNTildaEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include "ngp_algorithms/NodalGradAlgDriver.h"

namespace stk {
struct topology;
}

namespace sierra {
namespace kynema_ugf {

class Realm;
class LinearSystem;
class EquationSystems;

class AFTNTildaEquationSystem : public EquationSystem
{
public:
  AFTNTildaEquationSystem(EquationSystems& equationSystems);
  virtual ~AFTNTildaEquationSystem();

  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);

  void register_interior_algorithm(stk::mesh::Part* part);

  void register_inflow_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const InflowBoundaryConditionData& inflowBCData);

  void register_open_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const OpenBoundaryConditionData& openBCData);

  void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData);

  virtual void register_symmetry_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const SymmetryBoundaryConditionData& symmetryBCData);

  virtual void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();
  void solve_and_update();
  void update_and_clip();
  void predict_state();
  void assemble_nodal_gradient();
  void compute_effective_diff_flux_coeff();

  const bool managePNG_;

  ScalarFieldType* nTilda_;
  VectorFieldType* dntdx_;
  ScalarFieldType* nTmp_;
  ScalarFieldType* minDistanceToWall_;
  VectorFieldType* dwalldistdx_;
  ScalarFieldType* nDotV_;
  VectorFieldType* dnDotVdx_;
  ScalarFieldType* visc_;
  ScalarFieldType* tvisc_;
  ScalarFieldType* evisc_;
  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  ScalarNodalGradAlgDriver wallDistGradAlgDriver_;
  ScalarNodalGradAlgDriver nDotVGradAlgDriver_;

  std::unique_ptr<Algorithm> effDiffFluxAlg_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif