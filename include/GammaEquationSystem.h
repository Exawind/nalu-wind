#ifndef GammaEquationSystem_h
#define GammaEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>

#include "ngp_algorithms/NodalGradAlgDriver.h"
#include "ngp_algorithms/WallDistGradAlgDriver.h"
#include "ngp_algorithms/NDotVGradAlgDriver.h"

namespace stk{
struct topology;
}

namespace sierra{
namespace nalu{

class Realm;
class LinearSystem;
class EquationSystems;
class WallDistEquationSystem;


class GammaEquationSystem : public EquationSystem {

public:
  GammaEquationSystem(
      EquationSystems& equationSystems);

  virtual ~GammaEquationSystem();
  
  virtual void register_nodal_fields(stk::mesh::Part *part);

  void register_interior_algorithm(stk::mesh::Part *part);

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

  void initialize();

  void reinitialize_linear_system();

  void predict_state();

  void normalize_dwalldistdx();
  void compute_norm_dot_vel();

  void assemble_nodal_gradient();
  void assemble_walldist_gradient();
  void assemble_ndotv_gradient();
  void comp_eff_diff_coeff();

  const bool managePNG_;

  WallDistEquationSystem *walldistEqSys_;

  ScalarFieldType *gamma_;
  ScalarFieldType *gammaprod_;
  ScalarFieldType *gammasink_;
  ScalarFieldType *gammareth_;
  VectorFieldType *dGamdx_;
  VectorFieldType *dWallDistdx_;
  VectorFieldType *dNDotVdx_;
  ScalarFieldType *gamTmp_;
  ScalarFieldType *minDistanceToWall_;
  ScalarFieldType *NDotV_;
  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  ScalarNodalGradAlgDriver nodalGradAlgDriver_;
  ScalarWallDistGradAlgDriver walldistGradAlgDriver_;
  ScalarNDotVGradAlgDriver ndotvGradAlgDriver_;
  std::unique_ptr<Algorithm> effDiffFluxCoeffAlg_;

};

} // namespace nalu
} // namespace Sierra

#endif
