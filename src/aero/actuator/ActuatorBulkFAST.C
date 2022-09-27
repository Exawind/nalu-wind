// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorBulkFAST.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <aero/actuator/ActuatorFunctorsFAST.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

ActuatorMetaFAST::ActuatorMetaFAST(const ActuatorMeta& actMeta)
  : ActuatorMeta(actMeta),
    turbineNames_(numberOfActuators_),
    turbineOutputFileNames_(numberOfActuators_),
    maxNumPntsPerBlade_(0),
    epsilonTower_("epsilonTowerMeta", numberOfActuators_),
    epsilonHub_("epsilonHubMeta", numberOfActuators_),
    useUniformAziSampling_(
      "diskUseUniSample", is_disk() ? numberOfActuators_ : 0),
    nPointsSwept_("diskNumSwept", is_disk() ? numberOfActuators_ : 0),
    nBlades_("numTurbBlades", numberOfActuators_)
{
}

int
ActuatorMetaFAST::get_fast_index(
  fast::ActuatorNodeType type, int turbId, int index, int bladeNum) const
{
  return actuator_utils::get_fast_point_index(
    fastInputs_, turbId, nBlades_(turbId), type, index, bladeNum);
}

bool
ActuatorMetaFAST::is_disk()
{
  return actuatorType_ == ActuatorType::ActDiskFASTNGP;
}

ActuatorBulkFAST::ActuatorBulkFAST(
  const ActuatorMetaFAST& actMeta, double naluTimeStep)
  : ActuatorBulk(actMeta),
    turbineThrust_("turbineThrust", actMeta.numberOfActuators_),
    turbineTorque_("turbineTorque", actMeta.numberOfActuators_),
    hubLocations_("hubLocations", actMeta.numberOfActuators_),
    hubOrientation_("hubOrientations", actMeta.numberOfActuators_),
    orientationTensor_(
      "orientationTensor",
      actMeta.isotropicGaussian_ ? 0 : actMeta.numPointsTotal_),
    tStepRatio_(std::round(naluTimeStep / actMeta.fastInputs_.dtFAST))
{
  init_openfast(actMeta, naluTimeStep);
  init_epsilon(actMeta);
  RunActFastUpdatePoints(*this);
}

ActuatorBulkFAST::~ActuatorBulkFAST() { openFast_.end(); }

bool
ActuatorBulkFAST::is_tstep_ratio_admissable(
  const double fastTimeStep, const double naluTimeStep)
{
  const double stepCheck = std::abs(naluTimeStep / fastTimeStep - tStepRatio_);
  return stepCheck < 1e-12;
}

void
ActuatorBulkFAST::init_openfast(
  const ActuatorMetaFAST& actMeta, const double naluTimeStep)
{
  openFast_.setInputs(actMeta.fastInputs_);
  if (!is_tstep_ratio_admissable(actMeta.fastInputs_.dtFAST, naluTimeStep)) {
    throw std::runtime_error("ActuatorFAST: Ratio of Nalu's time step is not "
                             "an integral multiple of FAST time step.");
  } else {
    NaluEnv::self().naluOutputP0()
      << "Time step ratio  dtNalu/dtFAST: " << tStepRatio_ << std::endl;
  }

  const int nProcs = NaluEnv::self().parallel_size();
  const int nTurb = actMeta.numberOfActuators_;
  const int intDivision = nTurb / nProcs;
  const int remainder = actMeta.numberOfActuators_ % nProcs;
  const int nOffset = intDivision * nProcs;

  ThrowErrorMsgIf(
    remainder && intDivision,
    "nalu-wind can't process more turbines than ranks.");

  // assign turbines to processors uniformly
  for (int i = 0; i < intDivision; i++) {
    for (int j = 0; j < nProcs; j++) {
      openFast_.setTurbineProcNo(j + i * nProcs, j);
    }
  }
  for (int i = 0; i < remainder; i++) {
    openFast_.setTurbineProcNo(i + nOffset, i);
  }

  if (actMeta.fastInputs_.debug) {
    openFast_.init();
  } else {
    squash_fast_output(std::bind(&fast::OpenFAST::init, &openFast_));
  }
  /* TODO update/uncomment this check once openfast adds in a way
  to get the actual time step from fast::OpenFAST
  if (!is_tstep_ratio_admissable(openFast_.dtFAST, naluTimeStep)) {
    throw std::runtime_error("OpenFAST is using a different time step than "
                             "what was specified in the input deck. "
                             "Please check that your workflow is consistent "
                             "(restarts, FAST files, etc.");
  }
  */

  for (int i = 0; i < nTurb; ++i) {
    if (localTurbineId_ == openFast_.get_procNo(i)) {
      ThrowErrorMsgIf(
        actMeta.nBlades_(i) != openFast_.get_numBlades(i),
        "Mismatch in number of blades between OpenFAST and input deck."
        " InputDeck: " +
          std::to_string(actMeta.nBlades_(i)) +
          " OpenFAST: " + std::to_string(openFast_.get_numBlades(i)));
    }
  }
}

void
ActuatorBulkFAST::init_epsilon(const ActuatorMetaFAST& actMeta)
{
  // set epsilon and search radius

  epsilon_.modify_host();
  epsilonOpt_.modify_host();
  searchRadius_.modify_host();
  const int nTurb = openFast_.get_nTurbinesGlob();

  NaluEnv::self().naluOutputP0()
    << "Total Number of Actuator Points is: " << actMeta.numPointsTotal_
    << std::endl;

  for (int iTurb = 0; iTurb < nTurb; iTurb++) {
    if (openFast_.get_procNo(iTurb) == NaluEnv::self().parallel_rank()) {
      ThrowAssert(actMeta.numPointsTotal_ >= openFast_.get_numForcePts(iTurb));
      const int numForcePts = actMeta.numPointsTurbine_.h_view(iTurb);
      const int offset = turbIdOffset_.h_view(iTurb);
      auto epsilonChord =
        Kokkos::subview(actMeta.epsilonChord_.view_host(), iTurb, Kokkos::ALL);
      auto epsilonRef =
        Kokkos::subview(actMeta.epsilon_.view_host(), iTurb, Kokkos::ALL);
      auto epsilonTower =
        Kokkos::subview(actMeta.epsilonTower_.view_host(), iTurb, Kokkos::ALL);

      for (int np = 0; np < numForcePts; np++) {

        auto epsilonLocal =
          Kokkos::subview(epsilon_.view_host(), np + offset, Kokkos::ALL);
        auto epsilonOpt =
          Kokkos::subview(epsilonOpt_.view_host(), np + offset, Kokkos::ALL);

        switch (openFast_.getForceNodeType(iTurb, np)) {
        case fast::HUB: {
          // if epsilonHub hasn't already been set use model
          // of the wake (Martinez-Tossas PhD Thesis 2017)
          if (actMeta.epsilonHub_.h_view(iTurb, 0) <= 0) {
            float nac_cd = openFast_.get_nacelleCd(iTurb);
            // Compute epsilon only if drag coefficient is greater than zero
            if (nac_cd > 0) {
              float nac_area = openFast_.get_nacelleArea(iTurb);

              // This model is used to set the momentum thickness
              // of the wake (Martinez-Tossas PhD Thesis 2017)
              float tmpEps = std::sqrt(2.0 / M_PI * nac_cd * nac_area);
              for (int i = 0; i < 3; i++) {
                epsilonLocal(i) = tmpEps;
              }
            }
            // If no nacelle force just specify the standard value
            // (it will not be used)
            else {
              for (int i = 0; i < 3; i++) {
                epsilonLocal(i) = epsilonRef(i);
              }
            }
            for (int i = 0; i < 3; i++) {
              epsilonOpt(i) = epsilonLocal(i);
            }
          }
          break;
        }
        case fast::BLADE: {
          double chord = openFast_.getChord(np, iTurb);
          for (int i = 0; i < 3; i++) {
            // Define the optimal epsilon
            epsilonOpt(i) = epsilonChord(i) * chord;
            epsilonLocal(i) = std::max(epsilonOpt(i), epsilonRef(i));
          }
          break;
        }
        case fast::TOWER: {
          for (int i = 0; i < 3; i++) {
            epsilonLocal(i) = epsilonTower(i);
            epsilonOpt(i) = epsilonLocal(i);
          }
          break;
        }
        default:
          throw std::runtime_error("Actuator line model node type not valid");
          break;
        }

        for (int i = 0; i < 3; ++i) {
          ThrowAssertMsg(
            epsilonLocal(i) > 0.0,
            "Epsilon zero for point: " + std::to_string(np) + " index " +
              std::to_string(i));
        }

        // The radius of the searching. This is given in terms of
        //   the maximum of epsilon.x/y/z/.
        //
        // This is the length where the value of the Gaussian becomes
        // 0.1 % (1.0 / .001 = 1000) of the value at the center of the Gaussian
        searchRadius_.h_view(np + offset) =
          std::max(
            epsilonLocal(0), std::max(epsilonLocal(1), epsilonLocal(2))) *
          2.6282608848784661; // sqrt(log(1000))
      }
    }
  }
  actuator_utils::reduce_view_on_host(epsilon_.view_host());
  actuator_utils::reduce_view_on_host(epsilonOpt_.view_host());
  actuator_utils::reduce_view_on_host(searchRadius_.view_host());
  epsilon_.sync_host();
  epsilonOpt_.sync_host();
  searchRadius_.sync_host();
}

Kokkos::RangePolicy<ActuatorFixedExecutionSpace>
ActuatorBulkFAST::local_range_policy()
{
  auto rank = NaluEnv::self().parallel_rank();
  if (rank == openFast_.get_procNo(rank)) {
    const int offset = turbIdOffset_.h_view(rank);
    const int size = openFast_.get_numForcePts(rank);
    return Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(
      offset, offset + size);
  } else {
    return Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0, 0);
  }
}

void
ActuatorBulkFAST::interpolate_velocities_to_fast()
{
  openFast_.interpolateVel_ForceToVelNodes();

  if (openFast_.isTimeZero()) {
    if (openFast_.isDebug()) {
      openFast_.solution0();
    } else {
      squash_fast_output(std::bind(&fast::OpenFAST::solution0, &openFast_));
    }
  }
}

void
ActuatorBulkFAST::step_fast()
{
  if (openFast_.isDebug()) {
    for (int j = 0; j < tStepRatio_; j++) {
      openFast_.step();
    }
  } else {
    for (int j = 0; j < tStepRatio_; j++) {
      squash_fast_output(std::bind(&fast::OpenFAST::step, &openFast_));
    }
  }
}

bool
ActuatorBulkFAST::fast_is_time_zero()
{
  int localFastZero = (int)openFast_.isTimeZero();
  int globalFastZero = 0;
  MPI_Allreduce(
    &localFastZero, &globalFastZero, 1, MPI_INT, MPI_SUM,
    NaluEnv::self().parallel_comm());
  return globalFastZero > 0;
}

void
ActuatorBulkFAST::output_torque_info(stk::mesh::BulkData& stkBulk)
{
  Kokkos::parallel_for(
    "setUpTorqueCalc", hubLocations_.extent(0), ActFastSetUpThrustCalc(*this));

  actuator_utils::reduce_view_on_host(hubLocations_);
  actuator_utils::reduce_view_on_host(hubOrientation_);

  Kokkos::parallel_for(
    "computeTorque", coarseSearchElemIds_.extent(0),
    ActFastComputeThrust(*this, stkBulk));
  actuator_utils::reduce_view_on_host(turbineThrust_);
  actuator_utils::reduce_view_on_host(turbineTorque_);

  for (size_t iTurb = 0; iTurb < turbineThrust_.extent(0); iTurb++) {

    int processorId = openFast_.get_procNo(iTurb);

    if (NaluEnv::self().parallel_rank() == processorId) {
      auto thrust = Kokkos::subview(turbineThrust_, iTurb, Kokkos::ALL);
      auto torque = Kokkos::subview(turbineTorque_, iTurb, Kokkos::ALL);
      NaluEnv::self().naluOutput()
        << std::endl
        << "  Thrust[" << iTurb << "] = " << thrust(0) << " " << thrust(1)
        << " " << thrust(2) << " " << std::endl;
      NaluEnv::self().naluOutput()
        << "  Torque[" << iTurb << "] = " << torque(0) << " " << torque(1)
        << " " << torque(2) << " " << std::endl;

      std::vector<double> tmpThrust(3);
      std::vector<double> tmpTorque(3);

      openFast_.computeTorqueThrust(iTurb, tmpTorque, tmpThrust);

      NaluEnv::self().naluOutput()
        << "  Thrust ratio actual/correct = [" << thrust(0) / tmpThrust[0]
        << " " << thrust(1) / tmpThrust[1] << " " << thrust(2) / tmpThrust[2]
        << "] " << std::endl;
      NaluEnv::self().naluOutput()
        << "  Torque ratio actual/correct = [" << torque(0) / tmpTorque[0]
        << " " << torque(1) / tmpTorque[1] << " " << torque(2) / tmpTorque[2]
        << "] " << std::endl;
    }
  }
}

} // namespace nalu
} // namespace sierra
