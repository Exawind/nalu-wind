// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef MassFractionEquationSystem_h
#define MassFractionEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsedTypes.h>

// c++
#include <set>

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

class MassFractionEquationSystem : public EquationSystem {

public:

  MassFractionEquationSystem(
      EquationSystems& equationSystems,
      const int numMassFraction);
  virtual ~MassFractionEquationSystem();
  
  void register_nodal_fields(
      stk::mesh::Part *part);
  
  void register_interior_algorithm(
      stk::mesh::Part *part);

  void register_inflow_bc(
      stk::mesh::Part *part,
      const stk::topology &theTopo,
      const InflowBoundaryConditionData &inflowBCData);

  void register_open_bc(
      stk::mesh::Part *part,
      const stk::topology &theTopo,
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

  void predict_state();

  void set_current_mass_fraction(
      const int k);

  void copy_mass_fraction(
      const stk::mesh::FieldBase &fromField,
      const int fromFieldIndex,
      const stk::mesh::FieldBase &toField,
      const int toFieldIndex);

  void solve_and_update();
  void compute_nth_mass_fraction();

  bool system_is_converged();
  double provide_scaled_norm();
  double provide_norm();

  const bool managePNG_;

  const int numMassFraction_;
  
  GenericFieldType *massFraction_;
  ScalarFieldType *currentMassFraction_;
  VectorFieldType *dydx_;
  ScalarFieldType *yTmp_;
  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;

  AssembleNodalGradAlgorithmDriver *assembleNodalGradAlgDriver_;
  AlgorithmDriver *diffFluxCoeffAlgDriver_;
  
  bool isInit_;

  double nonLinearResidualSum_;
  double firstNonLinearResidualSum_;

  // hold all of the pairs of boundary condition data
  std::set<std::pair<stk::mesh::FieldBase *, stk::mesh::FieldBase *> > bcMassFractionSet_;
};

} // namespace nalu
} // namespace Sierra

#endif
