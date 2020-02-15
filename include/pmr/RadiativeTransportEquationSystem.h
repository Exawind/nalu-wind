// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef RadiativeTransportEquationSystem_h
#define RadiativeTransportEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsedTypes.h>

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class AlgorithmDriver;
class Realm;
class LinearSystem;
class EquationSystems;


class RadiativeTransportEquationSystem : public EquationSystem {

public:

  RadiativeTransportEquationSystem(
      EquationSystems& equationSystems,
      const int quadratureOrder,
      const bool activateScattering,
      const bool activateUpwind,
      const bool deactivateSucv,
      const bool externalCoupling);
  virtual ~RadiativeTransportEquationSystem();
  
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
      const stk::topology &theTopo,
      const WallBoundaryConditionData &wallBCData);

  void initialize();

  void predict_state();
  
  void solve_and_update();

  void set_current_ordinate_info(
      const int k);

  void initialize_intensity();
  void compute_bc_intensity();
  void compute_radiation_source();

  bool system_is_converged();
  double provide_scaled_norm();
  double provide_norm();

  void zero_out_fields();
  void zero_irradiation();
  
  void assemble_boundary_area();
  
  void assemble_fields();
  void assemble_irradiation();
  void normalize_irradiation();

  void compute_div_norm();
  
  void copy_ordinate_intensity(
      const ScalarFieldType &fromField,
      const ScalarFieldType &toField);
  
  void get_current_ordinate_info(
      double &weight,
      double *Sk) const;

  void get_current_ordinate(
      double *Sk) const;

  double get_stefan_boltzmann() const;
  
  ScalarFieldType *
  get_intensity() const;
  
  void create_quadrature_set();  

  const int quadratureOrder_;
  const bool activateScattering_;
  const bool activateUpwind_;
  const bool deactivateSucv_;
  const bool externalCoupling_;
  
  ScalarFieldType *intensity_;
  ScalarFieldType *currentIntensity_;
  ScalarFieldType *intensityBc_;
  ScalarFieldType *emissivity_;
  ScalarFieldType *transmissivity_;
  ScalarFieldType *environmentalT_;
  ScalarFieldType *iTmp_;
  ScalarFieldType *dualNodalVolume_;
  VectorFieldType *coordinates_;
  ScalarFieldType *temperature_;
  VectorFieldType *radiativeHeatFlux_;
  ScalarFieldType *divRadiativeHeatFlux_;
  ScalarFieldType *radiationSource_;
  ScalarFieldType *scalarFlux_;
  ScalarFieldType *scalarFluxOld_;
  ScalarFieldType *absorptionCoeff_;
  ScalarFieldType *scatteringCoeff_;
  VectorFieldType *edgeAreaVec_;
  ScalarFieldType *irradiation_;
  ScalarFieldType *bcTemperature_;
  ScalarFieldType *assembledBoundaryArea_;
  AlgorithmDriver *bcIntensityAlgDriver_;
  
  bool isInit_;
  int ordinateDirections_;

  // total set
  std::vector<double> Sn_;
  std::vector<double> weights_;

  // current set
  std::vector<double> currentSn_;
  double currentWeight_;
  double stefanBoltz_;
  double systemL2Norm_;
  double nonLinearResidualSum_;
  double firstNonLinearResidualSum_;

  // saved of mesh parts for interior and boundary
  std::vector<stk::mesh::Part *> interiorPartVec_;
  std::vector<stk::mesh::Part *> bcPartVec_;

};


} // namespace nalu
} // namespace Sierra

#endif
