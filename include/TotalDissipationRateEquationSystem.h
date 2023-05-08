// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TotalDissipationRateEquationSystem_h
#define TotalDissipationRateEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsedTypes.h>

#include "ngp_algorithms/NodalGradAlgDriver.h"

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

class Realm;
class LinearSystem;
class EquationSystems;

class TotalDissipationRateEquationSystem : public EquationSystem
{

public:
  TotalDissipationRateEquationSystem(EquationSystems& equationSystems);
  virtual ~TotalDissipationRateEquationSystem();

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

  virtual void register_non_conformal_bc(
    stk::mesh::Part* part, const stk::topology& theTopo);

  virtual void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();

  void predict_state();
  void assemble_nodal_gradient();
  void compute_effective_diff_flux_coeff();

  const bool managePNG_;

  ScalarFieldType* tdr_;
  VectorFieldType* dedx_;
  ScalarFieldType* eTmp_;
  ScalarFieldType* visc_;
  ScalarFieldType* tvisc_;
  ScalarFieldType* evisc_;
  ScalarFieldType* tdrWallBc_;
  ScalarFieldType* assembledWallTdr_;
  ScalarFieldType* assembledWallArea_;

  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  std::unique_ptr<Algorithm> effDiffFluxAlg_;
  std::unique_ptr<NgpAlgDriver> wallModelAlgDriver_;
};

} // namespace nalu
} // namespace sierra

#endif
