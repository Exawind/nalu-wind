// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ShearStressTransportEquationSystem_h
#define ShearStressTransportEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <KynemaUGFParsedTypes.h>
#include "ngp_algorithms/NodalGradAlgDriver.h"

namespace stk {
struct topology;
namespace mesh {
class Part;
}
} // namespace stk

namespace sierra {
namespace kynema_ugf {

class EquationSystems;
class AlgorithmDriver;
class TurbKineticEnergyEquationSystem;
class SpecificDissipationRateEquationSystem;
class GammaEquationSystem;

class ShearStressTransportEquationSystem : public EquationSystem
{

public:
  ShearStressTransportEquationSystem(EquationSystems& equationSystems);
  virtual ~ShearStressTransportEquationSystem();

  void load(const YAML::Node&) override;

  void initialize() override;

  void register_nodal_fields(const stk::mesh::PartVector& part_vec) override;

  void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData) override;

  void register_interior_algorithm(stk::mesh::Part* part) override;

  void register_inflow_bc(
    stk::mesh::Part* part,
    const stk::topology&,
    const InflowBoundaryConditionData&) override;
  void register_open_bc(
    stk::mesh::Part* part,
    const stk::topology&,
    const OpenBoundaryConditionData&) override;
  void register_symmetry_bc(
    stk::mesh::Part* part,
    const stk::topology&,
    const SymmetryBoundaryConditionData&) override;

  void solve_and_update() override;

  void initial_work() override;
  void post_external_data_transfer_work() override;
  void post_iter_work() final;
  void pre_iter_work() final;

  void clip_min_distance_to_wall();
  void compute_f_one_blending();
  void update_and_clip();
  void update_and_clip_gamma();
  void clip_sst(
    const stk::mesh::NgpMesh& ngpMesh,
    const stk::mesh::Selector& sel,
    stk::mesh::NgpField<double>& tke,
    stk::mesh::NgpField<double>& sdr);
  void clip_sst_gamma(
    const stk::mesh::NgpMesh& ngpMesh,
    const stk::mesh::Selector& sel,
    stk::mesh::NgpField<double>& gamma);

  TurbKineticEnergyEquationSystem* tkeEqSys_;
  SpecificDissipationRateEquationSystem* sdrEqSys_;
  GammaEquationSystem* gammaEqSys_;

  ScalarFieldType* tke_;
  ScalarFieldType* sdr_;
  ScalarFieldType* gamma_;
  ScalarFieldType* minDistanceToWall_;
  ScalarFieldType* fOneBlending_;
  ScalarFieldType* maxLengthScale_;

  bool isInit_;
  AlgorithmDriver* sstMaxLengthScaleAlgDriver_;

  // saved of mesh parts that are for wall bcs
  std::vector<stk::mesh::Part*> wallBcPart_;

  bool resetAMSAverages_;

  const double tkeMinValue_{1.0e-8};
  const double sdrMinValue_{1.0e-8};
  const double gammaMinValue_{0.02};
  const double gammaMaxValue_{1.0};

private:
  void register_wall_normal_gradient_bc(AlgorithmType, stk::mesh::Part*);
  void compute_wall_normal_gradients();

  VectorFieldType* dwalldistdx_{nullptr};
  ScalarFieldType* nDotV_{nullptr};
  VectorFieldType* dnDotVdx_{nullptr};
  ScalarNodalGradAlgDriver wallDistGradAlgDriver_;
  ScalarNodalGradAlgDriver nDotVGradAlgDriver_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif
