// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorFunctorsSimple.h>
#include <actuator/UtilitiesActuator.h>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <NaluEnv.h>
#include <FieldTypeDef.h>
#include "utils/LinearInterpolation.h"
#include <cmath>

namespace sierra {
namespace nalu {

InterpActuatorDensity::InterpActuatorDensity(
  ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk)
  : actBulk_(actBulk),
    stkBulk_(stkBulk),
    coordinates_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates")),
    density_(stkBulk_.mesh_meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "density"))
{
  actBulk_.density_.sync_host();
  actBulk_.density_.modify_host();
}

void
InterpActuatorDensity::operator()(int index) const
{
  auto rho = actBulk_.density_.view_host();
  auto localCoord = actBulk_.localCoords_;

  if (actBulk_.pointIsLocal_(index)) {

    stk::mesh::Entity elem = stkBulk_.get_entity(
      stk::topology::ELEMENT_RANK, actBulk_.elemContainingPoint_(index));

    const int nodesPerElem = stkBulk_.num_nodes(elem);

    // just allocate for largest expected size (hex27)
    double ws_coordinates[81], ws_density[81];

    actuator_utils::gather_field(
      3, &ws_coordinates[0], *coordinates_, stkBulk_.begin_nodes(elem),
      nodesPerElem);

    actuator_utils::gather_field_for_interp(
      1, &ws_density[0], *density_, stkBulk_.begin_nodes(elem), nodesPerElem);

    actuator_utils::interpolate_field(
      1, elem, stkBulk_, &(localCoord(index, 0)), &ws_density[0],
      &(rho(index)));
    rho(index) /= actBulk_.localParallelRedundancy_(index);
  }
}

ActSimpleUpdatePoints::ActSimpleUpdatePoints(ActuatorBulkSimple& actBulk, 
					     int numpts)
  : points_(helper_.get_local_view(actBulk.pointCentroid_)),
    offsets_(helper_.get_local_view(actBulk.turbIdOffset_)),
    turbId_(actBulk.localTurbineId_),
    numpoints_(numpts)
{
  helper_.touch_dual_view(actBulk.pointCentroid_);
}

void
ActSimpleUpdatePoints::operator()(int index) const
{

  ThrowAssert(turbId_ >= 0);
  const int pointId = index - offsets_(turbId_);
  auto point = Kokkos::subview(points_, index, Kokkos::ALL);

  // Uncomment this section if you need to update point locations
  // ------
  // std::vector<double> dx(3, 0.0);
  // double denom = (double)numpoints_;
  // for (int i=0; i<3; i++) {
  //   dx[i] = (p2_[i] - p1_[i])/denom; 
  // }
  // for (int i=0; i<3; i++) {
  //   point(i) = p1_[i] + 0.5*dx[i] + dx[i]*(float)pointId;
  // }
  // ------

}

ActSimpleAssignVel::ActSimpleAssignVel(ActuatorBulkSimple& actBulk)
  : velocity_(helper_.get_local_view(actBulk.velocity_)),
    density_(helper_.get_local_view(actBulk.density_)),
    points_(helper_.get_local_view(actBulk.pointCentroid_)),
    offset_(helper_.get_local_view(actBulk.turbIdOffset_)),
    turbId_(actBulk.localTurbineId_),
    debug_output_(actBulk.debug_output_)
{
}

void
ActSimpleAssignVel::operator()(int index) const
{

  const int pointId = index - offset_(turbId_);
  auto vel = Kokkos::subview(velocity_, index, Kokkos::ALL);
  auto rho = Kokkos::subview(density_, index);

  // Use this to double check the velocities and point positions
  auto point = Kokkos::subview(points_, index, Kokkos::ALL);
  if (debug_output_)
    NaluEnv::self().naluOutput() 
      << "Blade "<< turbId_  // LCCOUT
      << " pointId: " << pointId << std::scientific<< std::setprecision(5)
      << " point: "<<point(0)<<" "<<point(1)<<" "<<point(2)<<" "
      << " vel: "<<vel(0)<<" "<<vel(1)<<" "<<vel(2)<<" "
      << " rho: "<< *rho.data() 
      << std::endl;
  // Do nothing otherwise

}

ActSimpleComputeForce::ActSimpleComputeForce(ActuatorBulkSimple& actBulk,
					     const ActuatorMetaSimple& actMeta)
  : force_(helper_.get_local_view(actBulk.actuatorForce_)),
    velocity_(helper_.get_local_view(actBulk.velocity_)),
    density_(helper_.get_local_view(actBulk.density_)),
    offset_(helper_.get_local_view(actBulk.turbIdOffset_)),
    turbId_(actBulk.localTurbineId_),
    Npolartable(actMeta.polartable_size_.h_view(turbId_)),
    aoa_polartableDv_("aoa_polartable_Dv", actMeta.polartable_size_.h_view(turbId_)),
    cl_polartableDv_("cl_polartable_Dv", actMeta.polartable_size_.h_view(turbId_)),
    cd_polartableDv_("cd_polartable_Dv", actMeta.polartable_size_.h_view(turbId_)),
    Npts(actMeta.num_force_pts_blade_.h_view(turbId_)),
    twist_tableDv_("twist_table_Dv", actMeta.num_force_pts_blade_.h_view(turbId_)),
    elem_areaDv_("elem_area_Dv", actMeta.num_force_pts_blade_.h_view(turbId_)),
    debug_output_(actBulk.debug_output_)
{

  helper_.touch_dual_view(actBulk.actuatorForce_);
  if (NaluEnv::self().parallel_rank() == turbId_) {
    // Set up the polar table arrays
    for (size_t i=0; i<Npolartable; i++) {
      aoa_polartableDv_.h_view(i) = actMeta.aoa_polartableDv_.h_view(turbId_,i);
      cl_polartableDv_.h_view(i)  = actMeta.cl_polartableDv_.h_view(turbId_, i);
      cd_polartableDv_.h_view(i)  = actMeta.cd_polartableDv_.h_view(turbId_, i);
    }
    // Copy over the twist/area tables
    for (size_t i=0; i<Npts; i++) {
      twist_tableDv_.h_view(i) = actMeta.twist_tableDv_.h_view(turbId_, i);
      elem_areaDv_.h_view(i)   = actMeta.elem_areaDv_.h_view(turbId_, i);
    }
    
    // extract the directions
    for (int i=0; i<3; i++) {
      p1zeroalphadir[i] = actMeta.p1zeroalphadir_.h_view(turbId_, i);
      chordnormaldir[i] = actMeta.chordnormaldir_.h_view(turbId_, i);
      spandir[i]        = actMeta.spandir_.h_view(turbId_, i);
    }

  }
}

void
ActSimpleComputeForce::operator()(int index) const
{

  auto pointForce = Kokkos::subview(force_, index, Kokkos::ALL);
  const int localId = index - offset_(turbId_);

  auto vel     = Kokkos::subview(velocity_, index, Kokkos::ALL);
  auto density = Kokkos::subview(density_, index);

  if (NaluEnv::self().parallel_rank() == turbId_) {

  double twist = twist_tableDv_.h_view(localId); 

  double ws[3] = {vel(0), vel(1), vel(2)} ; // Total wind speed
 
  // Calculate the angle of attack (AOA)
  double alpha;
  double ws2D[3];
  AirfoilTheory2D::calculate_alpha(ws, p1zeroalphadir, 
				   spandir, chordnormaldir, twist, 
				   ws2D, alpha);

  // set up the polar tables
  std::vector<double> aoatable;
  std::vector<double> cltable;
  std::vector<double> cdtable;
  for (int i=0; i<Npolartable; i++) {
    aoatable.push_back(aoa_polartableDv_.h_view(i));
    cltable.push_back(cl_polartableDv_.h_view(i));
    cdtable.push_back(cd_polartableDv_.h_view(i));
  }

  // Calculate Cl and Cd
  double cl;
  double cd;
  utils::linear_interp(aoatable, cltable, alpha, cl);
  utils::linear_interp(aoatable, cdtable, alpha, cd);

  // Magnitude of wind speed
  double ws2Dnorm = sqrt(ws2D[0]*ws2D[0] + 
			 ws2D[1]*ws2D[1] +
			 ws2D[2]*ws2D[2]);
  
  // Calculate lift and drag forces
  double rho  = *density.data();
  double area = elem_areaDv_.h_view(localId); 
  double Q    = 0.5*rho*ws2Dnorm*ws2Dnorm;
  double lift = cl*Q*area;
  double drag = cd*Q*area;

  // Set the directions
  double ws2Ddir[3];  // Direction of drag force
  if (ws2Dnorm > 0.0) {
    ws2Ddir[0] = ws2D[0]/ws2Dnorm;
    ws2Ddir[1] = ws2D[1]/ws2Dnorm;
    ws2Ddir[2] = ws2D[2]/ws2Dnorm;
  } else {
    ws2Ddir[0] = 0.0; 
    ws2Ddir[1] = 0.0; 
    ws2Ddir[2] = 0.0; 
  }
  double liftdir[3];      // Direction of lift force
  if (ws2Dnorm > 0.0) {
    liftdir[0] = ws2Ddir[1]*spandir[2] - ws2Ddir[2]*spandir[1]; 
    liftdir[1] = ws2Ddir[2]*spandir[0] - ws2Ddir[0]*spandir[2]; 
    liftdir[2] = ws2Ddir[0]*spandir[1] - ws2Ddir[1]*spandir[0]; 
  } else {
    liftdir[0] = 0.0; 
    liftdir[1] = 0.0; 
    liftdir[2] = 0.0; 
  }

  // Set the pointForce
  pointForce(0) = -(lift*liftdir[0] + drag*ws2Ddir[0]);
  pointForce(1) = -(lift*liftdir[1] + drag*ws2Ddir[1]);
  pointForce(2) = -(lift*liftdir[2] + drag*ws2Ddir[2]);

  if (debug_output_)
    NaluEnv::self().naluOutput() 
      << "Blade "<< turbId_  // LCCOUT 
      << " pointId: " << localId << std::setprecision(5)
      << " alpha: "<<alpha
      << " ws2D: "<<ws2D[0]<<" "<<ws2D[1]<<" "<<ws2D[2]<<" "
      << " Cl, Cd: "<<cl<<" "<<cd
      << " lift, drag = "<<lift<<" "<<drag
      << std::endl;
  }
}

void 
AirfoilTheory2D::calculate_alpha(
    double ws[],                 
    const double zeroalphadir[], 
    const double spandir[],      
    const double chordnormaldir[],
    double twist, 
    double ws2D[],   
    double &alpha) 
{
  // Project WS onto 2D plane defined by zeroalpahdir and chordnormaldir
  double WSspan = ws[0]*spandir[0] + ws[1]*spandir[1] + ws[2]*spandir[2];
  ws2D[0] = ws[0] - WSspan*spandir[0];
  ws2D[1] = ws[1] - WSspan*spandir[1];
  ws2D[2] = ws[2] - WSspan*spandir[2];

  // Project WS2D onto zeroalphadir and chordnormaldir
  double WStan = 
    ws2D[0]*zeroalphadir[0] + 
    ws2D[1]*zeroalphadir[1] +  
    ws2D[2]*zeroalphadir[2] ;
  
  double WSnormal = 
    ws2D[0]*chordnormaldir[0] + 
    ws2D[1]*chordnormaldir[1] + 
    ws2D[2]*chordnormaldir[2] ;
  
  double alphaNoTwist = atan2(WSnormal, WStan)*180.0/M_PI;

  alpha = alphaNoTwist + twist;  
}

ActSimpleSetUpThrustCalc::ActSimpleSetUpThrustCalc(ActuatorBulkSimple& actBulk)
  : actBulk_(actBulk)
{
}

void
ActSimpleSetUpThrustCalc::operator()(int index) const
{
  auto thrust = Kokkos::subview(actBulk_.turbineThrust_, index, Kokkos::ALL);

  for (int i = 0; i < 3; i++) {
    thrust(i) = 0.0;
  }
}

void
ActSimpleComputeThrustInnerLoop::operator()(
  const uint64_t pointId,
  const double* nodeCoords,
  double* sourceTerm,
  const double dual_vol,
  const double scvIp) const
{

  auto offsets = actBulk_.turbIdOffset_.view_host();

  if (NaluEnv::self().parallel_rank()<actBulk_.num_blades_) {
    int turbId = NaluEnv::self().parallel_rank();
    auto thrust = Kokkos::subview(actBulk_.turbineThrust_, turbId, Kokkos::ALL);

  double r[3], rPerpShaft[3], forceTerm[3];

  for (int i = 0; i < 3; i++) {
    forceTerm[i] = sourceTerm[i]*scvIp;
    thrust(i) += forceTerm[i];
  }

  }
}

void
ActSimpleSpreadForceWhProjInnerLoop::preloop()
{
  actBulk_.actuatorForce_.sync_host();
}

void
ActSimpleSpreadForceWhProjInnerLoop::operator()(
  const uint64_t pointId,
  const double* nodeCoords,
  double* sourceTerm,
  const double dual_vol,
  const double scvIp) const
{

  auto pointCoords =
    Kokkos::subview(actBulk_.pointCentroid_.view_host(), pointId, Kokkos::ALL);

  auto pointForce =
    Kokkos::subview(actBulk_.actuatorForce_.view_host(), pointId, Kokkos::ALL);

  auto epsilon =
    Kokkos::subview(actBulk_.epsilon_.view_host(), pointId, Kokkos::ALL);

  auto orientation = Kokkos::subview(
    actBulk_.orientationTensor_.view_host(), pointId, Kokkos::ALL);

  double distance[3]={0, 0, 0};
  double projectedDistance[3]={0, 0, 0};
  double projectedForce[3]={0, 0, 0};

  actuator_utils::compute_distance(
    3, nodeCoords, pointCoords.data(), &distance[0]);

  // transform distance from Cartesian to blade coordinate system
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      projectedDistance[i] += distance[j] * orientation(i+j*3);
    }
  }

  const double gauss = actuator_utils::Gaussian_projection(
    3, &projectedDistance[0], epsilon.data());

  for (int j = 0; j < 3; j++) {
    projectedForce[j] = gauss * pointForce(j);
  }

  for (int j = 0; j < 3; j++) {
    sourceTerm[j] += projectedForce[j] * scvIp / dual_vol;
  }

}

} /* namespace nalu */
} /* namespace sierra */
