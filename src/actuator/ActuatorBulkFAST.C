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
    epsilon_("epsilonMeta", numberOfActuators_),
    epsilonChord_("epsilonChordMeta", numberOfActuators_),
    epsilonTower_("epsilonTowerMeta", numberOfActuators_)
{
}

ActuatorBulkFAST::ActuatorBulkFAST(
  const ActuatorMetaFAST& actMeta, stk::mesh::BulkData& stkBulk)
  : ActuatorBulk(actMeta, stkBulk),
    turbIdOffset_("offsetsForTurbine", actMeta.numberOfActuators_),
    epsilonOpt_("epsilonOptimal", actMeta.numberOfActuators_)
{
  openFast_.setInputs(actMeta.fastInputs_);
  // TODO(psakiev) copy functionality of ActuatorFAST::setup can do in
  // meta/parse

  TOUCH_DUAL_VIEW(turbIdOffset_, Kokkos::HostSpace)

  const int numTurbs = actMeta.numberOfActuators_;

  for (int i = 1; i < numTurbs; i++) {
    turbIdOffset_.h_view(i) =
      turbIdOffset_.h_view(i - 1) + actMeta.numPointsTurbine_.h_view(i - 1);
  }

  const int nProcs = NaluEnv::self().parallel_size();
  const int nTurb = actMeta.numberOfActuators_;
  const int intDivision = nTurb / nProcs;
  const int remainder = actMeta.numberOfActuators_ % nProcs;

  // assign turbines to processors uniformly
  for (int i = 0; i < intDivision; i++) {
    openFast_.setTurbineProcNo(i, i);
  }
  for (int i = 0; i < remainder; i++) {
    openFast_.setTurbineProcNo(intDivision + i, i);
  }

  openFast_.init();

  if (openFast_.isDryRun()) {
    return;
  }

  // set epsilon and radius
  // The node ordering (from FAST) is
  // Node 0 - Hub node
  // Blade 1 nodes
  // Blade 2 nodes
  // Blade 3 nodes
  // Tower nodes
  TOUCH_DUAL_VIEW(epsilon_, Kokkos::HostSpace)
  TOUCH_DUAL_VIEW(epsilonOpt_, Kokkos::HostSpace)
  TOUCH_DUAL_VIEW(searchRadius_, Kokkos::HostSpace)
  for (int iTurb = 0; iTurb < numTurbs; iTurb++) {
    if (openFast_.get_procNo(iTurb) == NaluEnv::self().parallel_rank()) {
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
    } else {
      NaluEnv::self().naluOutput() << "Proc " << NaluEnv::self().parallel_rank()
                                   << " glob iTurb " << iTurb << std::endl;
    }
  }
}

ActuatorBulkFAST::~ActuatorBulkFAST() { openFast_.end(); }

} // namespace nalu
} // namespace sierra
