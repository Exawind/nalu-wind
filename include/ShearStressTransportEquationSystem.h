/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ShearStressTransportEquationSystem_h
#define ShearStressTransportEquationSystem_h

#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>

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
  void post_adapt_work();

  void clip_min_distance_to_wall();
  void compute_f_one_blending();
  void compute_f_ABLBlending();
  void update_and_clip();

  TurbKineticEnergyEquationSystem *tkeEqSys_;
  SpecificDissipationRateEquationSystem *sdrEqSys_;

  ScalarFieldType *tke_;
  ScalarFieldType *sdr_;
  ScalarFieldType *minDistanceToWall_;
  ScalarFieldType *fOneBlending_;
  ScalarFieldType *fABLBlending_;
  ScalarFieldType *maxLengthScale_;

  bool isInit_;
  AlgorithmDriver *sstMaxLengthScaleAlgDriver_;

  // saved of mesh parts that are for wall bcs
  std::vector<stk::mesh::Part *> wallBcPart_;
     
};

} // namespace nalu
} // namespace Sierra

#endif
