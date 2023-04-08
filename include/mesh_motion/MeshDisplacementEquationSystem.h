// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MeshDisplacementEquationSystem_h
#define MeshDisplacementEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsedTypes.h>

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

class AlgorithmDriver;
class AssembleNodalGradUAlgorithmDriver;
class Realm;
class LinearSystem;

class MeshDisplacementEquationSystem : public EquationSystem
{

public:
  MeshDisplacementEquationSystem(
    EquationSystems& equationSystems,
    const bool activateMass,
    const bool deformWrtModelCoords);
  virtual ~MeshDisplacementEquationSystem();

  void initial_work();

  void register_nodal_fields(stk::mesh::Part* part);

  void
  register_element_fields(stk::mesh::Part* part, const stk::topology& theTopo);

  void register_interior_algorithm(stk::mesh::Part* part);

  void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData);

  void register_overset_bc();

  void initialize();
  void reinitialize_linear_system();

  void predict_state();
  void solve_and_update();
  void compute_current_coordinates();
  void compute_div_mesh_velocity();

  const bool activateMass_;
  const bool deformWrtModelCoords_;
  bool isInit_;
  VectorFieldType* meshDisplacement_;
  VectorFieldType* meshVelocity_;
  GenericFieldType* dvdx_;
  ScalarFieldType* divV_;
  VectorFieldType* coordinates_;
  VectorFieldType* currentCoordinates_;
  ScalarFieldType* dualNodalVolume_;
  ScalarFieldType* density_;
  ScalarFieldType* lameMu_;
  ScalarFieldType* lameLambda_;
  VectorFieldType* dxTmp_;

  AssembleNodalGradUAlgorithmDriver* assembleNodalGradAlgDriver_;
};

} // namespace nalu
} // namespace sierra

#endif
