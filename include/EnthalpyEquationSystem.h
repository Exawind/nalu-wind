// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef EnthalpyEquationSystem_h
#define EnthalpyEquationSystem_h

#include <memory>

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsedTypes.h>

#include "ngp_algorithms/NodalGradAlgDriver.h"
#include "ngp_algorithms/EnthalpyEffDiffFluxCoeffAlg.h"

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class AlgorithmDriver;
class Realm;
class AssembleWallHeatTransferAlgorithmDriver;
class LinearSystem;
class EquationSystems;
class ProjectedNodalGradientEquationSystem;
class TemperaturePropAlgorithm;

class EnthalpyEquationSystem : public EquationSystem {

public:

  EnthalpyEquationSystem(
    EquationSystems& equationSystems,
    const double minT,
    const double maxT,
    const bool outputClippingDiag);
  virtual ~EnthalpyEquationSystem();
  
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

  virtual void register_abltop_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const ABLTopBoundaryConditionData &abltopBCData);

  virtual void register_non_conformal_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo);

  virtual void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();

  virtual void register_initial_condition_fcn(
      stk::mesh::Part *part,
      const std::map<std::string, std::string> &theNames,
      const std::map<std::string, std::vector<double> > &theParams);

  void predict_state();
  
  void solve_and_update();
  void post_iter_work_dep();
  void extract_temperature();
  void post_converged_work();
  void initial_work();
  
  void temperature_bc_setup(
    UserData userData,
    stk::mesh::Part *part,
    ScalarFieldType *temperatureBc,
    ScalarFieldType *enthalpyBc,
    const bool isInterface = false,
    const bool copyBcVal = true);
  
  void manage_projected_nodal_gradient(
    EquationSystems& eqSystems);
  void compute_projected_nodal_gradient();

  const double minimumT_;
  const double maximumT_;

  const bool managePNG_;
  const bool outputClippingDiag_;

  ScalarFieldType *enthalpy_;
  ScalarFieldType *temperature_;
  VectorFieldType *dhdx_;
  ScalarFieldType *hTmp_;
  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  ScalarFieldType *thermalCond_;
  ScalarFieldType *specHeat_;
  ScalarFieldType *divQ_;
  ScalarFieldType *pOld_;

  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  std::unique_ptr<EnthalpyEffDiffFluxCoeffAlg> diffFluxCoeffAlg_;
  AssembleWallHeatTransferAlgorithmDriver *assembleWallHeatTransferAlgDriver_;
  
  bool pmrCouplingActive_;
  bool lowSpeedCompressActive_;

  ProjectedNodalGradientEquationSystem *projectedNodalGradEqs_;

  bool isInit_;

  std::vector<TemperaturePropAlgorithm *> enthalpyFromTemperatureAlg_;
  std::vector<Algorithm *> bdf2CopyStateAlg_;

  // bc enthalpy
  std::vector<TemperaturePropAlgorithm *> bcEnthalpyFromTemperatureAlg_;
  std::vector<Algorithm *> bcCopyStateAlg_;
  
};


} // namespace nalu
} // namespace Sierra

#endif
