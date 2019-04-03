/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef LowMachEquationSystem_h
#define LowMachEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class AlgorithmDriver;
class Realm;
class AssembleNodalGradAlgorithmDriver;
class AssembleNodalGradPAlgorithmDriver;
class AssembleNodalGradUAlgorithmDriver;
class MomentumEquationSystem;
class ContinuityEquationSystem;
class ComputeMdotAlgorithmDriver;
class LinearSystem;
class ProjectedNodalGradientEquationSystem;
class SurfaceForceAndMomentAlgorithmDriver;

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
  
  virtual void initialize();

  virtual void register_nodal_fields(
    stk::mesh::Part *part);

  virtual void register_edge_fields(
    stk::mesh::Part *part);
 
  virtual void register_element_fields(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_interior_algorithm(
    stk::mesh::Part *part);

  virtual void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const OpenBoundaryConditionData &openBCData);

  virtual void register_surface_pp_algorithm(
       const PostProcessingData &theData,
       stk::mesh::PartVector &partVector);

  virtual void register_initial_condition_fcn(
      stk::mesh::Part *part,
      const std::map<std::string, std::string> &theNames,
      const std::map<std::string, std::vector<double> > &theParams);

  virtual void pre_iter_work();
  virtual void solve_and_update();
  virtual void post_adapt_work();

  virtual void predict_state();

  void project_nodal_velocity();

  void post_converged_work();

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

  virtual void initial_work();

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
    const stk::topology &theTopo,
    const InflowBoundaryConditionData &inflowBCData);

  virtual void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const OpenBoundaryConditionData &openBCData);

  virtual void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const WallBoundaryConditionData &wallBCData);
    
  virtual void register_symmetry_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
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
  
  virtual void predict_state();

  void compute_wall_function_params();

  virtual void manage_projected_nodal_gradient(
     EquationSystems& eqSystems);
   virtual void compute_projected_nodal_gradient();

  virtual void save_diagonal_term(
    const std::vector<stk::mesh::Entity>&,
    const std::vector<int>&,
    const std::vector<double>&
  );

  virtual void save_diagonal_term(
    unsigned,
    const stk::mesh::Entity*,
    const SharedMemView<const double**>&
  );

  virtual void save_diagonal_term(
    unsigned,
    const ngp::Mesh::ConnectedNodes&,
    const SharedMemView<const double**>&
  );

  virtual void assemble_and_solve(
    stk::mesh::FieldBase *deltaSolution);

  const bool managePNG_;

  VectorFieldType *velocity_;
  GenericFieldType *dudx_;

  VectorFieldType *coordinates_;
  VectorFieldType *uTmp_;

  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  ScalarFieldType* Udiag_{nullptr};
  
  AssembleNodalGradUAlgorithmDriver *assembleNodalGradAlgDriver_;
  AlgorithmDriver *diffFluxCoeffAlgDriver_;
  AlgorithmDriver *tviscAlgDriver_;
  AlgorithmDriver *cflReyAlgDriver_;
  AlgorithmDriver *wallFunctionParamsAlgDriver_;

  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;

  double firstPNGResidual_;

  // saved of mesh parts that are not to be projected
  std::vector<stk::mesh::Part *> notProjectedPart_;
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

  AssembleNodalGradPAlgorithmDriver *assembleNodalGradPAlgDriver_;
  ComputeMdotAlgorithmDriver *computeMdotAlgDriver_;
  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;
};

} // namespace nalu
} // namespace Sierra

#endif
