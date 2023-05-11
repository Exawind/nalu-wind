// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef VolumeOfFluidEquationSystem_h
#define VolumeOfFluidEquationSystem_h

#include <memory>

#include "EquationSystem.h"
#include "FieldTypeDef.h"
#include "NaluParsedTypes.h"
#include "AMSAlgDriver.h"

#include "ngp_algorithms/NodalGradAlgDriver.h"
#include "ngp_algorithms/WallFricVelAlgDriver.h"
#include "ngp_algorithms/EffDiffFluxCoeffAlg.h"
#include "ngp_algorithms/CourantReAlgDriver.h"

#include "stk_mesh/base/NgpMesh.hpp"

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

class AlgorithmDriver;
class Realm;
class LinearSystem;
class ProjectedNodalGradientEquationSystem;
class NgpAlgDriver;

class VolumeOfFluidEquationSystem : public EquationSystem
{

public:
  VolumeOfFluidEquationSystem(EquationSystems& equationSystems);
  virtual ~VolumeOfFluidEquationSystem();

  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);
  virtual void register_edge_fields(const stk::mesh::PartVector& part_vec);
  virtual void register_element_fields(
    const stk::mesh::PartVector& part_vec, const stk::topology& theTopo);

  virtual void register_interior_algorithm(stk::mesh::Part* part);

  virtual void register_inflow_bc(
    stk::mesh::Part* part,
    const stk::topology& partTopo,
    const InflowBoundaryConditionData& inflowBCData);

  virtual void register_open_bc(
    stk::mesh::Part* part,
    const stk::topology& partTopo,
    const OpenBoundaryConditionData& openBCData);

  virtual void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData);

  virtual void register_symmetry_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const SymmetryBoundaryConditionData& symmetryBCData);

  virtual void register_abltop_bc(
    stk::mesh::Part* part,
    const stk::topology& partTopo,
    const ABLTopBoundaryConditionData& ablTopBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part* part, const stk::topology& theTopo);

  virtual void register_overset_bc();

  virtual void initialize();
  virtual void reinitialize_linear_system();

  virtual void register_initial_condition_fcn(
    stk::mesh::Part* part,
    const std::map<std::string, std::string>& theNames,
    const std::map<std::string, std::vector<double>>& theParams);

  virtual void manage_projected_nodal_gradient(EquationSystems& eqSystems);
  virtual void compute_projected_nodal_gradient();

  virtual void solve_and_update();

  const bool managePNG_;
  ScalarFieldType* volumeOfFluid_;
  VectorFieldType* dvolumeOfFluiddx_;
  ScalarFieldType* vofTmp_;

  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  ProjectedNodalGradientEquationSystem* projectedNodalGradEqs_;
  bool isInit_;
};

} // namespace nalu
} // namespace sierra

#endif
