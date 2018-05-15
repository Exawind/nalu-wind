/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef HeatCondEquationSystem_h
#define HeatCondEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>

#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class Realm;
class AssembleNodalGradAlgorithmDriver;
class AlgorithmDriver;
class EquationSystems;
class ProjectedNodalGradientEquationSystem;

class HeatCondEquationSystem : public EquationSystem {

public:
  HeatCondEquationSystem(
    EquationSystems& equationSystems);
  virtual ~HeatCondEquationSystem();

  void manage_png(
    EquationSystems& eqSystems);

  void register_nodal_fields(
    stk::mesh::Part *part);

  void register_edge_fields(
    stk::mesh::Part *part);

  void register_element_fields(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  void register_interior_algorithm(
    stk::mesh::Part *part);

  void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &partTopo,
    const WallBoundaryConditionData &wallBCData);
 
  virtual void register_non_conformal_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_overset_bc();

  virtual void register_initial_condition_fcn(
      stk::mesh::Part *part,
      const std::map<std::string, std::string> &theNames,
      const std::map<std::string, std::vector<double> > &theParams);

  void solve_and_update();
  void compute_projected_nodal_gradient();

  void initialize();
  void reinitialize_linear_system();
 
  void predict_state();
  
  virtual void load(const YAML::Node & node)
  {
    EquationSystem::load(node);
  }


  // allow equation system to manage a projected nodal gradient
  const bool managePNG_;

  ScalarFieldType *temperature_;
  VectorFieldType *dtdx_;
  ScalarFieldType *tTmp_;
  ScalarFieldType *dualNodalVolume_;
  VectorFieldType *coordinates_;
  ScalarFieldType *exact_temperature_;
  VectorFieldType *exact_dtdx_;
  VectorFieldType *exact_laplacian_;
  
  ScalarFieldType *density_;
  ScalarFieldType *specHeat_;
  ScalarFieldType *thermalCond_;

  VectorFieldType *edgeAreaVec_;
 
  AssembleNodalGradAlgorithmDriver *assembleNodalGradAlgDriver_;
  bool isInit_;
  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;
};

} // namespace nalu
} // namespace Sierra

#endif
