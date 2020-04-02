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
#include <NaluParsedTypes.h>

namespace stk{
struct topology;
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class EquationSystems;
class AlgorithmDriver;
class TurbKineticEnergyEquationSystem;
class SpecificDissipationRateEquationSystem;

class ShearStressTransportEquationSystem : public EquationSystem {

public:

  ShearStressTransportEquationSystem(
    EquationSystems& equationSystems);
  virtual ~ShearStressTransportEquationSystem();

  virtual void load(const YAML::Node&);

  virtual void initialize();

  virtual void register_nodal_fields(
    stk::mesh::Part *part);

  virtual void register_wall_bc(
    stk::mesh::Part *part,
    const stk::topology &theTopo,
    const WallBoundaryConditionData &wallBCData);

  virtual void register_interior_algorithm(
					   stk::mesh::Part *part );

  virtual void solve_and_update();

  void initial_work();

  void clip_min_distance_to_wall();
  void compute_f_one_blending();
  void update_and_clip();

  TurbKineticEnergyEquationSystem *tkeEqSys_;
  SpecificDissipationRateEquationSystem *sdrEqSys_;

  ScalarFieldType *tke_;
  ScalarFieldType *sdr_;
  ScalarFieldType *minDistanceToWall_;
  ScalarFieldType *fOneBlending_;
  ScalarFieldType *maxLengthScale_;

  bool isInit_;
  AlgorithmDriver *sstMaxLengthScaleAlgDriver_;

  // saved of mesh parts that are for wall bcs
  std::vector<stk::mesh::Part *> wallBcPart_;

  bool resetTAMSAverages_;     
};

} // namespace nalu
} // namespace Sierra

#endif
