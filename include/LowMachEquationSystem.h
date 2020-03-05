// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef LowMachEquationSystem_h
#define LowMachEquationSystem_h

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

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class AlgorithmDriver;
class Realm;
class MomentumEquationSystem;
class ContinuityEquationSystem;
class LinearSystem;
class ProjectedNodalGradientEquationSystem;
class SurfaceForceAndMomentAlgorithmDriver;
class MdotAlgDriver;
class NgpAlgDriver;

/** Low-Mach formulation of the Navier-Stokes Equations
 *
 *  This class is a thin-wrapper around sierra::nalu::ContinuityEquationSystem
 *  and sierra::nalu::MomentumEquationSystem that orchestrates the interactions
 *  between the velocity and the pressure Possion solves in the
 *  LowMachEquationSystem::solve_and_update method.
 */
class LowMachEquationSystem : public EquationSystem {

public:

  LowMachEquationSystem (
    EquationSystems& equationSystems,
    const bool elementContinuityEqs);
  virtual ~LowMachEquationSystem();

  virtual void load(const YAML::Node&);

  virtual void initialize();

  virtual void register_nodal_fields(
    stk::mesh::Part *part);

  virtual void register_edge_fields(
    stk::mesh::Part *part);
 
  virtual void register_element_fields(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const OpenBoundaryConditionData &openBCData);

  virtual void register_initial_condition_fcn(
      stk::mesh::Part *part,
      const std::map<std::string, std::string> &theNames,
      const std::map<std::string, std::vector<double> > &theParams);

  virtual void pre_iter_work();
  virtual void solve_and_update();

  virtual void predict_state();

  void project_nodal_velocity();

  void post_converged_work();

  virtual void post_iter_work();

  const bool elementContinuityEqs_; /* allow for mixed element/edge for continuity */
  MomentumEquationSystem *momentumEqSys_;
  ContinuityEquationSystem *continuityEqSys_;

  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  ScalarFieldType *dualNodalVolume_;
  VectorFieldType *edgeAreaVec_;

  SurfaceForceAndMomentAlgorithmDriver *surfaceForceAndMomentAlgDriver_;
  std::vector<int> xyBCType_;

  bool isInit_;

};

/** Representation of the Momentum conservation equations in 2-D and 3-D
 *
 */
class MomentumEquationSystem : public EquationSystem {

public:

  MomentumEquationSystem(
    EquationSystems& equationSystems);
  virtual ~MomentumEquationSystem();

  virtual void initial_work() override;
  virtual void pre_timestep_work() override;

  virtual void register_nodal_fields(
    stk::mesh::Part *part) override;

  virtual void register_edge_fields(
    stk::mesh::Part *part) override;

  virtual void register_element_fields(
    stk::mesh::Part *part,
    const stk::topology &theTopo) override;

  virtual void register_interior_algorithm(
    stk::mesh::Part *part) override;

  virtual void register_inflow_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const InflowBoundaryConditionData &inflowBCData) override;

  virtual void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const OpenBoundaryConditionData &openBCData) override;

  virtual void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const WallBoundaryConditionData &wallBCData) override;
    
  virtual void register_symmetry_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const SymmetryBoundaryConditionData &symmetryBCData) override;

  virtual void register_abltop_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const ABLTopBoundaryConditionData &ablTopBCData) override;

  virtual void register_non_conformal_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo) override;

  virtual void register_overset_bc() override;

  virtual void initialize() override;
  virtual void reinitialize_linear_system() override;
  
  virtual void predict_state() override;

  void compute_wall_function_params();

  virtual void manage_projected_nodal_gradient(
     EquationSystems& eqSystems);
   virtual void compute_projected_nodal_gradient();

  virtual void save_diagonal_term(
    const std::vector<stk::mesh::Entity>&,
    const std::vector<int>&,
    const std::vector<double>&
  ) override;

  virtual void save_diagonal_term(
    unsigned,
    const stk::mesh::Entity*,
    const SharedMemView<const double**>&
  ) override;

  virtual void save_diagonal_term(
    unsigned,
    const stk::mesh::NgpMesh::ConnectedNodes&,
    const SharedMemView<const double**,DeviceShmem>&
  ) override;

  virtual void assemble_and_solve(
    stk::mesh::FieldBase *deltaSolution) override;

  void compute_turbulence_parameters();

  const bool managePNG_;

  VectorFieldType *velocity_;
  GenericFieldType *dudx_;

  VectorFieldType *coordinates_;
  VectorFieldType *uTmp_;

  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  ScalarFieldType *iddesRansIndicator_;

  VectorNodalGradAlgDriver nodalGradAlgDriver_;
  WallFricVelAlgDriver wallFuncAlgDriver_;
  NgpAlgDriver dynPressAlgDriver_;
  std::unique_ptr<EffDiffFluxCoeffAlg> diffFluxCoeffAlg_{nullptr};
  std::unique_ptr<Algorithm> tviscAlg_{nullptr};
  std::unique_ptr<Algorithm> pecletAlg_{nullptr};
  std::unique_ptr<Algorithm> ablWallNodeMask_ {nullptr};

  CourantReAlgDriver cflReAlgDriver_;
  std::unique_ptr<AMSAlgDriver> AMSAlgDriver_{nullptr};

  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;

  double firstPNGResidual_;

  bool RANSAblBcApproach_{false};

  // saved of mesh parts that are not to be projected
  std::vector<stk::mesh::Part *> notProjectedPart_;
  std::array<std::vector<stk::mesh::Part*>,3> notProjectedDir_;

  ScalarFieldType* get_diagonal_field() override { return Udiag_; }

private:
  ScalarFieldType* Udiag_;
};

class ContinuityEquationSystem : public EquationSystem {

public:

  ContinuityEquationSystem(
    EquationSystems& equationSystems,
    const bool elementContinuityEqs);
  virtual ~ContinuityEquationSystem();

  virtual void register_nodal_fields(
    stk::mesh::Part *part);

  virtual void register_edge_fields(
    stk::mesh::Part *part);

  virtual void register_element_fields(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_interior_algorithm(
    stk::mesh::Part *part);

  virtual void register_inflow_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const InflowBoundaryConditionData &inflowBCData);

  virtual void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const OpenBoundaryConditionData &openBCData);

  virtual void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const WallBoundaryConditionData &wallBCData);
    
  virtual void register_symmetry_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const SymmetryBoundaryConditionData &symmetryBCData);

  virtual void register_abltop_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const ABLTopBoundaryConditionData &ablTopBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_overset_bc();

  virtual void initialize();
  virtual void reinitialize_linear_system();    
  
  virtual void register_initial_condition_fcn(
      stk::mesh::Part *part,
      const std::map<std::string, std::string> &theNames,
      const std::map<std::string, std::vector<double> > &theParams);

  virtual void manage_projected_nodal_gradient(
    EquationSystems& eqSystems);
  virtual void compute_projected_nodal_gradient();

  virtual void create_constraint_algorithm(stk::mesh::FieldBase*);
  
  const bool elementContinuityEqs_;
  const bool managePNG_;
  ScalarFieldType *pressure_;
  VectorFieldType *dpdx_;
  ScalarFieldType *massFlowRate_;
  VectorFieldType *coordinates_;

  ScalarFieldType *pTmp_;

  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  std::unique_ptr<MdotAlgDriver> mdotAlgDriver_;
  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;
};

} // namespace nalu
} // namespace Sierra

#endif
