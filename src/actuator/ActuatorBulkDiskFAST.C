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

namespace sierra
{
namespace nalu
{

ActuatorBulkDiskFAST::ActuatorBulkDiskFAST(ActuatorMetaFAST& actMeta, double naluTimeStep):
  ActuatorBulkFAST(actMeta, naluTimeStep)
{
  resize_arrays();

}

void ActuatorBulkDiskFAST::compute_swept_point_count(ActuatorMetaFAST& actMeta){

  ActFixScalarInt nAddedPoints("nAddedPoints",openFast_.get_nTurbinesGlob());

  actMeta.numPointsTurbine_.sync_host();
  actMeta.numPointsTurbine_.modify_host();

  for(int iTurb=0; iTurb<openFast_.get_nTurbinesGlob(); ++iTurb){
    if(localTurbineId_==openFast_.get_procNo(iTurb)){
      const int nBlades = openFast_.get_numBlades(iTurb);

      if(actMeta.useUniformAziSampling_(iTurb)){
        nAddedPoints(iTurb) = actMeta.nPointsSwept_(iTurb)*nBlades*openFast_.get_numForcePtsBlade(iTurb);
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
        }
        const double radius = locator.get_radius(0);
        // even radial spacing minus the blades
        nAddedPoints(iTurb)+=std::max(static_cast<int>(2.0 * M_PI*radius/dR)-nBlades, 0);
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

void ActuatorBulkDiskFAST::resize_arrays()
{
  epsilon_.resize(30);
}

} /* namespace nalu */
} /* namespace sierra */
