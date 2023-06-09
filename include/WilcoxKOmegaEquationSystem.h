// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef WilcoxKOmegaEquationSystem_h
#define WilcoxKOmegaEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsedTypes.h>

namespace stk {
struct topology;
namespace mesh {
class Part;
}
} // namespace stk

namespace sierra {
namespace nalu {

class EquationSystems;
class AlgorithmDriver;
class TurbKineticEnergyEquationSystem;
class SpecificDissipationRateEquationSystem;

class WilcoxKOmegaEquationSystem : public EquationSystem
{

public:
  WilcoxKOmegaEquationSystem(EquationSystems& equationSystems);
  virtual ~WilcoxKOmegaEquationSystem();

  virtual void load(const YAML::Node&);

  virtual void initialize();

  virtual void register_nodal_fields(const stk::mesh::PartVector& part_vec);

  virtual void register_wall_bc(
    stk::mesh::Part* part,
    const stk::topology& theTopo,
    const WallBoundaryConditionData& wallBCData);

  virtual void register_interior_algorithm(stk::mesh::Part* part);

  virtual void solve_and_update();

  void initial_work();
  virtual void post_external_data_transfer_work();
  virtual void post_iter_work();

  void clip_min_distance_to_wall();
  void update_and_clip();
  void clip_ko(
    const stk::mesh::NgpMesh& ngpMesh,
    const stk::mesh::Selector& sel,
    stk::mesh::NgpField<double>& tke,
    stk::mesh::NgpField<double>& sdr);

  TurbKineticEnergyEquationSystem* tkeEqSys_;
  SpecificDissipationRateEquationSystem* sdrEqSys_;

  ScalarFieldType* tke_;
  ScalarFieldType* sdr_;
  ScalarFieldType* minDistanceToWall_;

  bool isInit_;

  // saved of mesh parts that are for wall bcs
  std::vector<stk::mesh::Part*> wallBcPart_;

  bool resetAMSAverages_;

  const double tkeMinValue_{1.0e-8};
  const double sdrMinValue_{1.0e-8};
};

} // namespace nalu
} // namespace sierra

#endif
