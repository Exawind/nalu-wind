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
  for(int iTurb=0; iTurb<openFast_.get_nTurbinesGlob(); ++iTurb){
    if(localTurbineId_==openFast_.get_procNo(iTurb)){
      if(actMeta.useUniformAziSampling_(iTurb)){
        nAddedPoints(iTurb) = actMeta.nPointsSwept_(iTurb);
      }
      else{
        // compute radii and dr
        // divide radius by dr and sum
      }
    }
  }
  actuator_utils::reduce_view_on_host(nAddedPoints);
}

void ActuatorBulkDiskFAST::resize_arrays()
{
  epsilon_.resize(30);
}

} /* namespace nalu */
} /* namespace sierra */
