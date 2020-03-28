// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBulkFAST.h>
#include <actuator/UtilitiesActuator.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

ActuatorMetaFAST::ActuatorMetaFAST(const ActuatorMeta& actMeta)
  : ActuatorMeta(actMeta),
    turbineNames_(numberOfActuators_),
    turbineOutputFileNames_(numberOfActuators_),
    filterLiftLineCorrection_(false),
    isotropicGaussian_(false),
    maxNumPntsPerBlade_(0),
    epsilon_("epsilonMeta", numberOfActuators_),
    epsilonChord_("epsilonChordMeta", numberOfActuators_),
    epsilonTower_("epsilonTowerMeta", numberOfActuators_),
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
    epsilonOpt_("epsilonOptimal", actMeta.numPointsTotal_),
    orientationTensor_(
      "orientationTensor",
      actMeta.isotropicGaussian_ ? 0 : actMeta.numPointsTotal_),
    localTurbineId_(
      NaluEnv::self().parallel_rank() >= actMeta.numberOfActuators_
        ? -1
        : NaluEnv::self().parallel_rank()), // assign 1 turbine per rank for now
    tStepRatio_(naluTimeStep / actMeta.fastInputs_.dtFAST)
{
  init_openfast(actMeta, naluTimeStep);
  init_epsilon(actMeta);
}

ActuatorBulkFAST::~ActuatorBulkFAST() { openFast_.end(); }

void
ActuatorBulkFAST::init_openfast(
  const ActuatorMetaFAST& actMeta, double naluTimeStep)
{
  openFast_.setInputs(actMeta.fastInputs_);

  if (
    std::abs(naluTimeStep - tStepRatio_ * actMeta.fastInputs_.dtFAST) <
    0.001) { // TODO: Fix
    // arbitrary number
    // 0.001
    NaluEnv::self().naluOutputP0()
      << "Time step ratio  dtNalu/dtFAST: " << tStepRatio_ << std::endl;
  } else {
    throw std::runtime_error("ActuatorFAST: Ratio of Nalu's time step is not "
                             "an integral multiple of FAST time step");
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

  openFast_.init();

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
  // set epsilon and radius

  epsilon_.modify_host();
  epsilonOpt_.modify_host();
  searchRadius_.modify_host();
  const int nTurb = openFast_.get_nTurbinesGlob();

  for (int iTurb = 0; iTurb < nTurb; iTurb++) {
    if (openFast_.get_procNo(iTurb) == NaluEnv::self().parallel_rank()) {
      ThrowAssert(actMeta.numPointsTotal_ >= openFast_.get_numForcePts(iTurb));
      const int numForcePts = openFast_.get_numForcePts(iTurb);
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
        searchRadius_.h_view(np + offset) =
          std::max(
            epsilonLocal(0), std::max(epsilonLocal(1), epsilonLocal(2))) *
          sqrt(log(1.0 / 0.001));
      }
    } else {
      NaluEnv::self().naluOutput() << "Proc " << NaluEnv::self().parallel_rank()
                                   << " glob iTurb " << iTurb << std::endl;
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
  if (!openFast_.isDryRun()) {
    openFast_.interpolateVel_ForceToVelNodes();

    if (openFast_.isTimeZero()) {
      openFast_.solution0();
    }
  }
}

void
ActuatorBulkFAST::step_fast()
{
  if (!openFast_.isDryRun()) {
    for (int j = 0; j < tStepRatio_; j++) {
      openFast_.step();
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
ActuatorBulkFAST::output_torque_info()
{
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
