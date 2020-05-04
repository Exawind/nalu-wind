// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBulkSimple.h>
#include <actuator/UtilitiesActuator.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

ActuatorMetaSimple::ActuatorMetaSimple(const ActuatorMeta& actMeta)
  : ActuatorMeta(actMeta),
    filterLiftLineCorrection_(false),
    isotropicGaussian_(false),
    epsilon_("epsilonMeta", numberOfActuators_),
    epsilonChord_("epsilonChordMeta", numberOfActuators_),
    num_force_pts_blade_("numForcePtsBladeMeta", numberOfActuators_),
    p1_("p1Meta", numberOfActuators_),
    p2_("p2Meta", numberOfActuators_),
    p1zeroalphadir_("p1zeroalphadirMeta", numberOfActuators_),
    chordnormaldir_("chordnormaldirMeta", numberOfActuators_),
    spandir_("spandirMeta", numberOfActuators_),
    max_num_force_pts_blade_(0),
    max_polartable_size_(0),
    polartable_size_("polartablesizeMeta", numberOfActuators_)
{
}

ActuatorBulkSimple::ActuatorBulkSimple(
  const ActuatorMetaSimple& actMeta, double naluTimeStep)
  : ActuatorBulk(actMeta),
    turbineThrust_("turbineThrust", actMeta.numberOfActuators_),
    epsilonOpt_("epsilonOptimal", actMeta.numPointsTotal_),
    orientationTensor_(
      "orientationTensor",
      actMeta.isotropicGaussian_ ? 0 : actMeta.numPointsTotal_),
    num_force_pts_blade_("numForcePtsBladeBulk", actMeta.numberOfActuators_),
    num_blades_(actMeta.numberOfActuators_),
    debug_output_(actMeta.debug_output_),
    assignedProc_("assignedProcBulk", actMeta.numberOfActuators_),
    localTurbineId_(
      NaluEnv::self().parallel_rank() >= actMeta.numberOfActuators_
        ? -1
      : NaluEnv::self().parallel_rank()) // assign 1 turbine per rank for now Used to be ? -1
{
  // Allocate blades to turbines
  const int nProcs = NaluEnv::self().parallel_size();
  const int nTurb = actMeta.numberOfActuators_;
  const int intDivision = nTurb / nProcs;
  const int remainder = actMeta.numberOfActuators_ % nProcs;

  if (actMeta.debug_output_) 
    NaluEnv::self().naluOutputP0() << " nProcs: " << nProcs 
				   << " nTurb:  " << nTurb
				   << " intDiv: " << intDivision
				   << " remain: " << remainder
				   << std::endl; // LCCOUT

  if (remainder && intDivision)  // this doesn't work for nProcs=1
    throw std::runtime_error(" ERRORXX: more blades than ranks");
  if (nTurb > nProcs) 
    throw std::runtime_error(" ERROR: more blades than ranks");

  for (int i=0; i<nTurb; i++) {
    assignedProc_.h_view(i) = i;
    NaluEnv::self().naluOutputP0() << " Turbine#: " << i
				   << " Proc#: " << assignedProc_.h_view(i) <<std::endl;

  }


  // Set up num_force_pts_blade_
  for (int i = 0; i <actMeta.numberOfActuators_; ++i) {
    num_force_pts_blade_.h_view(i) = actMeta.num_force_pts_blade_.h_view(i);
  }
  // Double check offsets
  if (actMeta.debug_output_) 
    for (int i = 0; i <actMeta.numberOfActuators_; ++i) {
      NaluEnv::self().naluOutputP0() << "Offset blade: " << i << " "
				     << turbIdOffset_.h_view(i) 
				     << " num_force_pts: "
				     << num_force_pts_blade_.h_view(i)
				     << std::endl; //LCCOUT
    }
  init_epsilon(actMeta);
  init_points(actMeta);
  init_orientation(actMeta);
  NaluEnv::self().naluOutputP0() << "Done ActuatorBulkSimple Init "
				 << std::endl; // LCCOUT
}

ActuatorBulkSimple::~ActuatorBulkSimple() { 
}

void
ActuatorBulkSimple::init_epsilon(const ActuatorMetaSimple& actMeta)
{
  // set epsilon and radius

  epsilon_.modify_host();
  epsilonOpt_.modify_host();
  searchRadius_.modify_host();

  const int nBlades = actMeta.n_simpleblades_;
  for (int iBlade = 0; iBlade<nBlades; iBlade++) {
    // LCC test this for non-isotropic
    if (NaluEnv::self().parallel_rank()==assignedProc_.h_view(iBlade)) { 
      const int numForcePts = actMeta.num_force_pts_blade_.h_view(iBlade);
      const int offset = turbIdOffset_.h_view(iBlade);      
      auto epsilonChord =
        Kokkos::subview(actMeta.epsilonChord_.view_host(), iBlade, Kokkos::ALL);
      auto epsilonRef =
        Kokkos::subview(actMeta.epsilon_.view_host(), iBlade, Kokkos::ALL);
      for (int np = 0; np < numForcePts; np++) {
        auto epsilonLocal =
          Kokkos::subview(epsilon_.view_host(), np + offset, Kokkos::ALL);
        auto epsilonOpt =
          Kokkos::subview(epsilonOpt_.view_host(), np + offset, Kokkos::ALL);

	double chord = actMeta.chord_tableDv_.h_view(iBlade, np); 
	for (int i = 0; i < 3; i++) {
	  // Define the optimal epsilon
	  epsilonOpt(i) = epsilonChord(i) * chord;
	  epsilonLocal(i) = std::max(epsilonOpt(i), epsilonRef(i));
	}
        // The radius of the searching. This is given in terms of
        //   the maximum of epsilon.x/y/z/.
        //
        // This is the length where the value of the Gaussian becomes
        // 0.1 % (1.0 / .001 = 1000) of the value at the center of the Gaussian
        searchRadius_.h_view(np + offset) =
          std::max(
            epsilonLocal(0), std::max(epsilonLocal(1), epsilonLocal(2))) *
          sqrt(log(1.e3));

      } // loop over np
    }
  } // loop over iBlade

  actuator_utils::reduce_view_on_host(epsilon_.view_host());
  actuator_utils::reduce_view_on_host(epsilonOpt_.view_host());
  actuator_utils::reduce_view_on_host(searchRadius_.view_host());
  epsilon_.sync_host();
  epsilonOpt_.sync_host();
  searchRadius_.sync_host();
}

// Initializes the point coordinates
void
ActuatorBulkSimple::init_points(const ActuatorMetaSimple& actMeta)
{
  pointCentroid_.modify_host();

  const int nBlades = actMeta.n_simpleblades_;
  for (int iBlade = 0; iBlade<nBlades; iBlade++) {
    if (NaluEnv::self().parallel_rank()==assignedProc_.h_view(iBlade)) { 
      const int numForcePts = actMeta.num_force_pts_blade_.h_view(iBlade);
      const int offset = turbIdOffset_.h_view(iBlade);      
      const double denom = (double)numForcePts;

      // Get p1 and p2 and dx for blade geometry
      double p1[3];
      double p2[3];
      double dx[3];
      for (int j=0; j<3; j++) { 
	p1[j] = actMeta.p1_.h_view(iBlade, j);
	p2[j] = actMeta.p2_.h_view(iBlade, j);
	dx[j] = (p2[j] - p1[j])/denom; 
      }

      // set every pointCentroid
      for (int np = 0; np < numForcePts; np++) {
        auto pointLocal =
          Kokkos::subview(pointCentroid_.view_host(), np + offset, Kokkos::ALL);

	for (int i=0; i<3; i++) {
	  pointLocal(i) = p1[i] + 0.5*dx[i] + dx[i]*(double)np;
	}

	NaluEnv::self().naluOutput() 
	  << "Blade "<< iBlade  // LCCOUT
	  << " pointId: " << np << std::scientific<< std::setprecision(5)
	  << " point: "<<pointLocal(0)<<" "<<pointLocal(1)<<" "<<pointLocal(2)
	  << std::endl;

      }// loop over np
    }
  } // loop over iBlade
  actuator_utils::reduce_view_on_host(pointCentroid_.view_host());
  pointCentroid_.sync_host();
}

// Initializes the orientation matrices
void
ActuatorBulkSimple::init_orientation(const ActuatorMetaSimple& actMeta)
{
  // Bail out if this is isotropic
  if (actMeta.isotropicGaussian_) return;

  orientationTensor_.modify_host();
  const int nBlades = actMeta.n_simpleblades_;
  for (int iBlade = 0; iBlade<nBlades; iBlade++) {
    if (NaluEnv::self().parallel_rank()==assignedProc_.h_view(iBlade)) { 
      const int numForcePts = actMeta.num_force_pts_blade_.h_view(iBlade);
      const int offset = turbIdOffset_.h_view(iBlade);      

      // set every pointCentroid
      for (int np = 0; np < numForcePts; np++) {
	auto orientation = Kokkos::subview(
            orientationTensor_.view_host(), np + offset, Kokkos::ALL);
	// set orientation tensor to identity
	orientation(0) = 1.0;
	orientation(1) = 0.0;
	orientation(2) = 0.0;
	orientation(3) = 1.0;
	orientation(4) = 0.0;
	orientation(5) = 0.0;
	orientation(6) = 1.0;
      }

    }
  } // loop over iBlade
  actuator_utils::reduce_view_on_host(orientationTensor_.view_host());
  orientationTensor_.sync_host();
}

Kokkos::RangePolicy<ActuatorFixedExecutionSpace>
ActuatorBulkSimple::local_range_policy()
{
  auto rank = NaluEnv::self().parallel_rank();
  if (rank < num_blades_) {
    const int offset = turbIdOffset_.h_view(rank);
    const int size   = num_force_pts_blade_.h_view(rank); 
    return Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(
      offset, offset + size);
  } else {
    return Kokkos::RangePolicy<ActuatorFixedExecutionSpace>(0, 0);
  }
}


void
ActuatorBulkSimple::zero_open_fast_views()
{
  dvHelper_.touch_dual_view(actuatorForce_);
  dvHelper_.touch_dual_view(velocity_);
  dvHelper_.touch_dual_view(density_);
  Kokkos::deep_copy(dvHelper_.get_local_view(actuatorForce_),0.0);
  Kokkos::deep_copy(dvHelper_.get_local_view(velocity_),0.0);
  Kokkos::deep_copy(dvHelper_.get_local_view(density_),0.0);
 
  // Uncomment this functor if you want to update the point positions
  // -----------
  //dvHelper_.touch_dual_view(pointCentroid_);
  //Kokkos::deep_copy(dvHelper_.get_local_view(pointCentroid_),0.0);
    
}

} // namespace nalu
} // namespace sierra
