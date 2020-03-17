// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorBulkDiskFAST.h>
#include <actuator/UtilitiesActuator.h>
#include <actuator/ActuatorFunctorsFAST.h>

namespace sierra
{
namespace nalu
{

ActuatorBulkDiskFAST::ActuatorBulkDiskFAST(ActuatorMetaFAST& actMeta, double naluTimeStep):
  ActuatorBulkFAST(actMeta, naluTimeStep),
  numSweptCount_("numSweptCount", 0),
  numSweptOffset_("numSweptOffset",actMeta.numberOfActuators_)
{
  int nOffset =0;
  for(int iTurb=0; iTurb<actMeta.numberOfActuators_; iTurb++){
    numSweptOffset_(iTurb)=nOffset;
    nOffset+=actMeta.fastInputs_.globTurbineData[iTurb].numForcePtsBlade;
  }

  Kokkos::resize(numSweptCount_, nOffset);

  compute_swept_point_count(actMeta);
  resize_arrays(actMeta);
  compute_offsets(actMeta);
  init_epsilon(actMeta);
  Kokkos::parallel_for("InitActLinePoints", local_range_policy(actMeta), ActFastUpdatePoints(*this));
  initialize_swept_points(actMeta);
}

void ActuatorBulkDiskFAST::compute_swept_point_count(ActuatorMetaFAST& actMeta){

  ActFixScalarInt nAddedPoints("nAddedPoints",openFast_.get_nTurbinesGlob());

  actMeta.numPointsTurbine_.sync_host();
  actMeta.numPointsTurbine_.modify_host();

  for(int iTurb=0; iTurb<openFast_.get_nTurbinesGlob(); ++iTurb){
    if(NaluEnv::self().parallel_rank()==openFast_.get_procNo(iTurb)){
      const int nBlades = openFast_.get_numBlades(iTurb);

      if(actMeta.useUniformAziSampling_(iTurb)){
        nAddedPoints(iTurb) = actMeta.nPointsSwept_(iTurb)*nBlades*openFast_.get_numForcePtsBlade(iTurb);
        for(int i=0; i<nAddedPoints.extent_int(0); ++i){
          numSweptCount_(i+numSweptOffset_(iTurb)) = actMeta.nPointsSwept_(iTurb);
        }
      }
      else{
        // compute radii and dr
        Point p1 = actuator_utils::get_fast_point(openFast_,iTurb, fast::BLADE, 0, 0);
        Point p2 = actuator_utils::get_fast_point(openFast_,iTurb, fast::BLADE, 1, 0);
        double dR = 0.0;
        for(int i=0; i<3; ++i){
          dR+=std::pow(p2[i]-p1[i],2.0);
        }
        dR = std::sqrt(dR);

        actuator_utils::SweptPointLocator locator;

        // divide radius by dr and sum
        for(int i=0; i<openFast_.get_numForcePtsBlade(iTurb); ++i){
          for(int j=0; j<nBlades; ++j){
            locator.update_point_location(j, actuator_utils::get_fast_point(openFast_,iTurb, fast::BLADE, i,j));
          }
          const double radius = locator.get_radius(0);
          // even radial spacing minus the blades
          numSweptCount_(i+numSweptOffset_(iTurb))= std::max((int)(2.0*M_PI*radius/dR/nBlades)-1,0);
          nAddedPoints(iTurb)+=numSweptCount_(i+numSweptOffset_(iTurb))*nBlades;
        }
      }
    }
    else{
      nAddedPoints(iTurb)=0;
    }
  }

  actuator_utils::reduce_view_on_host(nAddedPoints);

  for(int i=0; i<nAddedPoints.extent_int(0); ++i){
    actMeta.numPointsTurbine_.h_view(i)+=nAddedPoints(i);
    actMeta.numPointsTotal_+=nAddedPoints(i);
  }

}

void ActuatorBulkDiskFAST::resize_arrays(const ActuatorMetaFAST& actMeta)
{
  const int newSize = actMeta.numPointsTotal_;
  pointCentroid_.resize(newSize);
  actuatorForce_.resize(newSize);
  epsilon_.resize(newSize);
  epsilonOpt_.resize(newSize);
  searchRadius_.resize(newSize);
  Kokkos::resize(localCoords_, newSize);
  Kokkos::resize(pointIsLocal_, newSize);
  Kokkos::resize(elemContainingPoint_, newSize);
}

void ActuatorBulkDiskFAST::initialize_swept_points(const ActuatorMetaFAST& actMeta){
  actuator_utils::SweptPointLocator pointLocator;

  pointCentroid_.modify_host();
  epsilon_.modify_host();
  epsilonOpt_.modify_host();
  searchRadius_.modify_host();

  for(int iTurb=0; iTurb<actMeta.numberOfActuators_; iTurb++){
    const int nForcePtsBlade = actMeta.fastInputs_.globTurbineData[iTurb].numForcePtsBlade;
    const int turbOffset = turbIdOffset_.h_view(iTurb);
    const int sweptOffset = numSweptOffset_(iTurb);
    const int nBlades = 3; //TODO(psakiev) catch this error, disk will only work with 3 blades to define circle

    for(int iB = 0; iB<nForcePtsBlade; iB++){
      for(int bN=0; bN<nBlades; bN++){
        const int indexB =
            actuator_utils::get_fast_point_index(actMeta.fastInputs_, iTurb, nBlades,fast::BLADE, iB, bN);

        auto pnt = Kokkos::subview(pointCentroid_.view_host(), indexB, Kokkos::ALL);
        pointLocator.update_point_location(bN,Point{pnt(0), pnt(1), pnt(2)});
      }

      const double dTheta= 2.0*M_PI/(nBlades*(numSweptCount_(iB+sweptOffset)+1));
      double theta = M_PI/3.0;
      for(int bN=0; bN<nBlades; bN++){
        const int indexB =
            actuator_utils::get_fast_point_index(actMeta.fastInputs_, iTurb, nBlades,fast::BLADE, iB, bN);
        for(int j=0; j<numSweptCount_(iB+sweptOffset); j++){
          theta+=dTheta;
          Point pointCoords = pointLocator(theta);
          const int sweepIndex = turbOffset+sweptOffset+bN*nBlades+j;
          searchRadius_.h_view(sweepIndex) = searchRadius_.h_view(indexB);
          for(int k=0; k<3; k++){
            pointCentroid_.h_view(sweepIndex,k)=pointCoords[k];
            epsilon_.h_view(sweepIndex,k)=epsilon_.h_view(indexB,k);
            epsilonOpt_.h_view(sweepIndex,k)=epsilonOpt_.h_view(indexB,k);
          }
          theta+=dTheta;
        }
      }
    }
  }
}

} /* namespace nalu */
} /* namespace sierra */
