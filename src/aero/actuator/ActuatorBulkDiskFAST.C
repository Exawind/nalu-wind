// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorBulkDiskFAST.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <aero/actuator/ActuatorFunctorsFAST.h>

namespace sierra {
namespace nalu {
// TODO(psakiev) convert disk points to geometric series
// TODO(psakiev) allow for anisotropic disk

ActuatorBulkDiskFAST::ActuatorBulkDiskFAST(
  ActuatorMetaFAST& actMeta, double naluTimeStep)
  : ActuatorBulkFAST(actMeta, naluTimeStep),
    numSweptCount_(
      "numSweptCount", actMeta.numberOfActuators_, actMeta.maxNumPntsPerBlade_),
    numSweptOffset_(
      "numSweptOffset", actMeta.numberOfActuators_, actMeta.maxNumPntsPerBlade_)
{

  ThrowErrorIf(!actMeta.is_disk());
  compute_swept_point_count(actMeta);
  resize_arrays(actMeta);
  Kokkos::parallel_for(
    "ZeroArrays",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
      0, actMeta.numPointsTotal_),
    [&](int index) {
      for (int j = 0; j < 3; j++) {
        pointCentroid_.h_view(index, j) = 0;
        epsilon_.h_view(index, j) = 0;
        epsilonOpt_.h_view(index, j) = 0;
      }
      searchRadius_.h_view(index) = 0;
    });
  compute_offsets(actMeta);
  init_epsilon(actMeta);
  RunActFastUpdatePoints(*this);
  initialize_swept_points(actMeta);
}

// update the swept points (points between ALM blades) and then turn the update
// flag to false indicating an update has already taken place
void
ActuatorBulkDiskFAST::update_ADM_points(const ActuatorMetaFAST& actMeta)
{
  // TODO work out a trigger for if the yaw has changed
  initialize_swept_points(actMeta);
  adm_points_need_updating = false;
}

void
ActuatorBulkDiskFAST::compute_swept_point_count(ActuatorMetaFAST& actMeta)
{

  ActFixScalarInt nAddedPoints("nAddedPoints", openFast_.get_nTurbinesGlob());

  actMeta.numPointsTurbine_.sync_host();
  actMeta.numPointsTurbine_.modify_host();

  for (int iTurb = 0; iTurb < openFast_.get_nTurbinesGlob(); ++iTurb) {
    if (NaluEnv::self().parallel_rank() == openFast_.get_procNo(iTurb)) {
      const int nBlades = openFast_.get_numBlades(iTurb);
      const int nbfp = openFast_.get_numForcePtsBlade(iTurb);

      if (actMeta.useUniformAziSampling_(iTurb)) {
        nAddedPoints(iTurb) = actMeta.nPointsSwept_(iTurb) * nBlades * nbfp;
        for (int i = 0; i < nbfp; ++i) {
          numSweptCount_(iTurb, i) = actMeta.nPointsSwept_(iTurb);
          numSweptOffset_(iTurb, i) =
            actMeta.nPointsSwept_(iTurb) * i * nBlades;
        }
      } else {
        // compute radii and dr
        Point p1 =
          actuator_utils::get_fast_point(openFast_, iTurb, fast::BLADE, 0, 0);
        Point p2 =
          actuator_utils::get_fast_point(openFast_, iTurb, fast::BLADE, 1, 0);
        double dR = 0.0;
        for (int i = 0; i < 3; ++i) {
          dR += std::pow(p2[i] - p1[i], 2.0);
        }
        dR = std::sqrt(dR);

        actuator_utils::SweptPointLocator locator;

        // divide radius by dr and sum
        for (int i = 0; i < openFast_.get_numForcePtsBlade(iTurb); ++i) {
          for (int j = 0; j < nBlades; ++j) {
            locator.update_point_location(
              j, actuator_utils::get_fast_point(
                   openFast_, iTurb, fast::BLADE, i, j));
          }
          const double radius = locator.get_radius(0);
          // even radial spacing minus the blades
          numSweptCount_(iTurb, i) =
            std::max((int)(2.0 * M_PI * radius / dR / nBlades) - 1, 0);
          // WARNING:: Not thread safe
          numSweptOffset_(iTurb, i) =
            i == 0 ? 0
                   : numSweptCount_(iTurb, i - 1) * nBlades +
                       numSweptOffset_(iTurb, i - 1);
          nAddedPoints(iTurb) += numSweptCount_(iTurb, i) * nBlades;
        }
      }
    } else {
      nAddedPoints(iTurb) = 0;
    }
  }

  actuator_utils::reduce_view_on_host(nAddedPoints);
  actuator_utils::reduce_view_on_host(numSweptCount_);
  actuator_utils::reduce_view_on_host(numSweptOffset_);

  for (int i = 0; i < nAddedPoints.extent_int(0); ++i) {
    actMeta.numPointsTurbine_.h_view(i) += nAddedPoints(i);
    actMeta.numPointsTotal_ += nAddedPoints(i);
  }
}

void
ActuatorBulkDiskFAST::resize_arrays(const ActuatorMetaFAST& actMeta)
{
  const int newSize = actMeta.numPointsTotal_;
  pointCentroid_.resize(newSize);
  actuatorForce_.resize(newSize);
  epsilon_.resize(newSize);
  // we don't need velocity to resize, but resize to match search/apply plumbing
  velocity_.resize(newSize);
  epsilonOpt_.resize(newSize);
  searchRadius_.resize(newSize);
  Kokkos::resize(localCoords_, newSize);
  Kokkos::resize(pointIsLocal_, newSize);
  Kokkos::resize(localParallelRedundancy_, newSize);
  Kokkos::resize(elemContainingPoint_, newSize);
  // resize fflc arrays as well.  This is a memory waste, but the most simple
  // way to make things consistent
  Kokkos::resize(relativeVelocity_, newSize);
  if (actMeta.useFLLC_) {
    Kokkos::resize(relativeVelocityMagnitude_, newSize);
    Kokkos::resize(liftForceDistribution_, newSize);
    Kokkos::resize(deltaLiftForceDistribution_, newSize);
    Kokkos::resize(epsilonOpt_, newSize);
    Kokkos::resize(fllc_, newSize);
  }
}

void
ActuatorBulkDiskFAST::initialize_swept_points(const ActuatorMetaFAST& actMeta)
{
  actuator_utils::SweptPointLocator pointLocator;

  pointCentroid_.modify_host();
  epsilon_.modify_host();
  epsilonOpt_.modify_host();
  searchRadius_.modify_host();

  for (int iTurb = 0; iTurb < actMeta.numberOfActuators_; iTurb++) {

    const int nForcePtsBlade =
      actMeta.fastInputs_.globTurbineData[iTurb].numForcePtsBlade;
    const int turbOffset = turbIdOffset_.h_view(iTurb);
    const int turbTotal = actMeta.numPointsTurbine_.h_view(iTurb);
    const int nForcePtsFast =
      1 + actMeta.get_fast_index(
            fast::TOWER, iTurb,
            actMeta.fastInputs_.globTurbineData[iTurb].numForcePtsTwr - 1);

    auto sweptOffset = Kokkos::subview(numSweptOffset_, iTurb, Kokkos::ALL);
    auto sweptCount = Kokkos::subview(numSweptCount_, iTurb, Kokkos::ALL);
    auto points = Kokkos::subview(
      pointCentroid_.view_host(), std::make_pair(turbOffset, turbTotal),
      Kokkos::ALL);
    auto epsilon = Kokkos::subview(
      epsilon_.view_host(), std::make_pair(turbOffset, turbTotal), Kokkos::ALL);
    auto searchRadius = Kokkos::subview(
      searchRadius_.view_host(), std::make_pair(turbOffset, turbTotal));

    for (int iB = 0; iB < nForcePtsBlade; iB++) {
      const int localOffset = nForcePtsFast + sweptOffset(iB);

      for (int bN = 0; bN < 3; bN++) {
        const int indexB = actMeta.get_fast_index(fast::BLADE, iTurb, iB, bN);

        auto pnt = Kokkos::subview(points, indexB, Kokkos::ALL);
        pointLocator.update_point_location(bN, Point{pnt(0), pnt(1), pnt(2)});
      }

      // periodic function has blades points at pi/3, pi, and 5*pi/3
      // this is due to the way the control points are defined
      const double dTheta = 2.0 * M_PI / (3 * (sweptCount(iB) + 1));
      double theta = M_PI / 3.0;
      // assume identical blades
      const int indexB = actMeta.get_fast_index(fast::BLADE, iTurb, iB, 0);

      for (int nB = 0; nB < 3; nB++) {
        for (int nS = 0; nS < sweptCount(iB); nS++) {
          theta += dTheta;
          Point coords = pointLocator(theta);
          const int localIndex = localOffset + nS + sweptCount(iB) * nB;
          for (int i = 0; i < 3; i++) {
            points(localIndex, i) = coords[i];
            epsilon(localIndex, i) = epsilon(indexB, i);
          }
          searchRadius(localIndex) = searchRadius(indexB);
        }
        theta += dTheta; // skip blade
      }
    }
  }
}

void
ActuatorBulkDiskFAST::spread_forces_over_disk(const ActuatorMetaFAST& actMeta)
{
  actuatorForce_.sync_host();
  actuatorForce_.modify_host();

  for (int iTurb = 0; iTurb < actMeta.numberOfActuators_; iTurb++) {
    const int nForcePtsBlade =
      actMeta.fastInputs_.globTurbineData[iTurb].numForcePtsBlade;
    const int turbOffset = turbIdOffset_.h_view(iTurb);
    const int turbTotal = actMeta.numPointsTurbine_.h_view(iTurb);
    const int nForcePtsFast =
      1 + actMeta.get_fast_index(
            fast::TOWER, iTurb,
            actMeta.fastInputs_.globTurbineData[iTurb].numForcePtsTwr - 1);

    auto sweptOffset = Kokkos::subview(numSweptOffset_, iTurb, Kokkos::ALL);
    auto sweptCount = Kokkos::subview(numSweptCount_, iTurb, Kokkos::ALL);
    auto points = Kokkos::subview(
      pointCentroid_.view_host(), std::make_pair(turbOffset, turbTotal),
      Kokkos::ALL);
    auto forces = Kokkos::subview(
      actuatorForce_.view_host(), std::make_pair(turbOffset, turbTotal),
      Kokkos::ALL);

    for (int iB = 0; iB < nForcePtsBlade; iB++) {
      std::array<double, 3> forceTemp = {{0.0, 0.0, 0.0}};
      const int avgCount = (sweptCount(iB) + 1) * 3;
      const int localOffset = nForcePtsFast + sweptOffset(iB);

      for (int nB = 0; nB < 3; nB++) {
        for (int i = 0; i < 3; i++) {
          forceTemp[i] +=
            forces(actMeta.get_fast_index(fast::BLADE, iTurb, iB, nB), i);
        }
      }

      for (int i = 0; i < 3; i++) {
        forceTemp[i] /= avgCount;
      }

      // replace blade forces with distributed force
      for (int nB = 0; nB < 3; nB++) {
        for (int i = 0; i < 3; i++) {
          forces(actMeta.get_fast_index(fast::BLADE, iTurb, iB, nB), i) =
            forceTemp[i];
        }
      }

      // fill swept points with distributed force
      for (int nS = 0; nS < sweptCount(iB); nS++) {
        for (int nB = 0; nB < 3; nB++) {
          const int localIndex = localOffset + nS + sweptCount(iB) * nB;
          for (int i = 0; i < 3; i++) {
            forces(localIndex, i) = forceTemp[i];
          }
        }
      }
    }
  }
}

} /* namespace nalu */
} /* namespace sierra */
