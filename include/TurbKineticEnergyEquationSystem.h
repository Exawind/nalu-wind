/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbKineticEnergyEquationSystem_h
#define TurbKineticEnergyEquationSystem_h

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
class LinearSystem;
class EquationSystems;
class ProjectedNodalGradientEquationSystem;

class TurbKineticEnergyEquationSystem : public EquationSystem {

public:

  TurbKineticEnergyEquationSystem(
    EquationSystems& equationSystems);
  virtual ~TurbKineticEnergyEquationSystem();

  virtual void register_nodal_fields(
    stk::mesh::Part *part);

  void register_interior_algorithm(
    stk::mesh::Part *part);
  
  void register_inflow_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const InflowBoundaryConditionData &inflowBCData);
  
  void register_open_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const OpenBoundaryConditionData &openBCData);

  void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const WallBoundaryConditionData &wallBCData);
  
  virtual void register_symmetry_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const SymmetryBoundaryConditionData &symmetryBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();
  
  void predict_state();
  
  void solve_and_update();
  void initial_work();

  void compute_effective_diff_flux_coeff();
  void compute_wall_model_parameters();
  void update_and_clip();

  void manage_projected_nodal_gradient(
    EquationSystems& eqSystems);
  void compute_projected_nodal_gradient();
  
  const bool managePNG_;

  ScalarFieldType *tke_;
  VectorFieldType *dkdx_;
  ScalarFieldType *kTmp_;
  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  
  AssembleNodalGradAlgorithmDriver *assembleNodalGradAlgDriver_;
  AlgorithmDriver *diffFluxCoeffAlgDriver_;
  AlgorithmDriver *wallFunctionTurbKineticEnergyAlgDriver_;
  const TurbulenceModel turbulenceModel_;

  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;

  bool isInit_;

};


} // namespace nalu
} // namespace Sierra

#endif
