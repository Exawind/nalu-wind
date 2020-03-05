// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBulkFAST.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

ActuatorMetaFAST::ActuatorMetaFAST(const ActuatorMeta& actMeta)
  : ActuatorMeta(actMeta),
    turbineNames_(numberOfActuators_),
    turbineOutputFileNames_(numberOfActuators_),
    filterLiftLineCorrection_(false),
    timeStepRatio_(1.0),
    epsilon_("epsilonMeta", numberOfActuators_),
    epsilonChord_("epsilonChordMeta", numberOfActuators_),
    epsilonTower_("epsilonTowerMeta", numberOfActuators_)
{
}

ActuatorBulkFAST::ActuatorBulkFAST(
  const ActuatorMetaFAST& actMeta, stk::mesh::BulkData& stkBulk)
  : ActuatorBulk(actMeta, stkBulk),
    epsilonOpt_("epsilonOptimal", actMeta.numberOfActuators_),
    localTurbineId_(NaluEnv::self().parallel_rank()>=actMeta.numberOfActuators_?-1:NaluEnv::self().parallel_rank()),
    tStepRatio_(actMeta.timeStepRatio_)
{
  openFast_.setInputs(actMeta.fastInputs_);

  const int nProcs = NaluEnv::self().parallel_size();
  const int nTurb = actMeta.numberOfActuators_;
  const int intDivision = nTurb / nProcs;
  const int remainder = actMeta.numberOfActuators_ % nProcs;
  const int nOffset = intDivision*nProcs;

  if(remainder && intDivision){
    ThrowErrorMsg("OpenFAST can't accept more turbines than ranks.");
  }

  // assign turbines to processors uniformly
  for (int i=0; i < intDivision; i++){
    for(int j=0; j<nProcs; j++){
      openFast_.setTurbineProcNo(j+i*nProcs, j);
    }
  }
  for (int i = 0; i < remainder; i++) {
    openFast_.setTurbineProcNo(i+nOffset, i);
  }

  openFast_.init();


  // set epsilon and radius
  // The node ordering (from FAST) is
  // Node 0 - Hub node
  // Blade 1 nodes
  // Blade 2 nodes
  // Blade 3 nodes
  // Tower nodes
  epsilon_.modify_host();
  epsilonOpt_.modify_host();
  searchRadius_.modify_host();

  for (int iTurb = 0; iTurb < nTurb; iTurb++) {
    if (openFast_.get_procNo(iTurb) == NaluEnv::self().parallel_rank()) {
      if (!openFast_.isDryRun()) {
        const int numForcePts = openFast_.get_numForcePts(iTurb);
        const int offset = turbIdOffset_.h_view(iTurb);
        const double* epsilonChord = &(actMeta.epsilonChord_.h_view(iTurb, 0));
        const double* epsilonRef = &(actMeta.epsilon_.h_view(iTurb, 0));
        const double* epsilonTower = &(actMeta.epsilonTower_.h_view(iTurb, 0));

        for (int np = 0; np < numForcePts; np++) {

          double chord = openFast_.getChord(np, iTurb);
          double* epsilonLocal = &(epsilon_.h_view(np + offset, 0));
          double* epsilonOpt = &(epsilonOpt_.h_view(np + offset, 0));

          switch (openFast_.getForceNodeType(iTurb, np)) {
          case fast::HUB: {
            float nac_cd = openFast_.get_nacelleCd(iTurb);
            // Compute epsilon only if drag coefficient is greater than zero
            if (nac_cd > 0) {
              float nac_area = openFast_.get_nacelleArea(iTurb);

              // This model is used to set the momentum thickness
              // of the wake (Martinez-Tossas PhD Thesis 2017)
              float tmpEps = std::sqrt(2.0 / M_PI * nac_cd * nac_area);
              for (int i = 0; i < 3; i++) {
                epsilonLocal[i] = tmpEps;
              }
            }
            // If no nacelle force just specify the standard value
            // (it will not be used)
            else {
              for (int i = 0; i < 3; i++) {
                epsilonLocal[i] = epsilonRef[i];
              }
            }
            for (int i = 0; i < 3; i++) {
              epsilonOpt[i] = epsilonLocal[i];
            }
            break;
          }
          case fast::BLADE: {
            for (int i = 0; i < 3; i++) {
              // Define the optimal epsilon
              epsilonOpt[i] = epsilonChord[i] * chord;
              epsilonLocal[i] = std::max(epsilonOpt[i], epsilonRef[i]);
            }
            break;
          }
          case fast::TOWER: {
            for (int i = 0; i < 3; i++) {
              epsilonLocal[i] = epsilonTower[i];
              epsilonOpt[i] = epsilonLocal[i];
            }
            break;
          }
          default:
            throw std::runtime_error("Actuator line model node type not valid");
            break;
          }

          // The radius of the searching. This is given in terms of
          //   the maximum of epsilon.x/y/z/.
          searchRadius_.h_view(iTurb) =
              std::max(
                epsilonLocal[0], std::max(epsilonLocal[1], epsilonLocal[2])) *
                sqrt(log(1.0 / 0.001));
        }
      }
      else {
      NaluEnv::self().naluOutput() << "Proc " << NaluEnv::self().parallel_rank()
                                   << " glob iTurb " << iTurb << std::endl;
      }
    }
  }
}

ActuatorBulkFAST::~ActuatorBulkFAST() { openFast_.end(); }

Kokkos::RangePolicy<ActuatorFixedExecutionSpace>
ActuatorBulkFAST::local_range_policy(const ActuatorMeta& actMeta){
  auto rank = NaluEnv::self().parallel_rank();
  if(rank<turbIdOffset_.extent_int(0)){
    const int offset = turbIdOffset_.h_view(rank);
    const int size = actMeta.numPointsTurbine_.h_view(rank);
    return Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(offset,offset+size);
  }
  else{
    return Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0,0);
  }
}


void ActuatorBulkFAST::step_fast(){
  if (!openFast_.isDryRun()) {

    openFast_.interpolateVel_ForceToVelNodes();

    if (openFast_.isTimeZero()) {
      openFast_.solution0();
    }

    for (int j = 0; j < tStepRatio_; j++){
      openFast_.step();
    }
  }
}

} // namespace nalu
} // namespace sierra
