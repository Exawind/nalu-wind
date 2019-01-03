/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/WindEnergyAuxFunction.h>
#include <PecletFunction.h>
#include <Realm.h>
#include <SolutionOptions.h>

// basic c++
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

WindEnergyAuxFunction::WindEnergyAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<std::string> theStringParams,
  Realm &realm) :
  AuxFunction(beginPos, endPos),
  omegaBlend_(1.0),
  tanhFunction_(NULL),
  omegaMM_(3,0.0),
  centroidMM_(3,0.0)
{
  // check for omega blending
  const std::string omegaName = "omega";
  if ( realm.get_tanh_functional_form(omegaName) == "tanh") {
    const double c1 = realm.get_tanh_trans(omegaName);
    const double c2 = realm.get_tanh_width(omegaName);
    tanhFunction_ = new TanhFunction<double>(c1, c2);
  }

  if (theStringParams.size() < 1 )
    throw std::runtime_error("Wind_energy user function requires at least one string parameter");

  // get yaml mesh motion node
  const YAML::Node& meshMotionNode = realm.solutionOptions_->meshMotionNode_;
  const int numMotion = meshMotionNode.size();

  // declare temporary variables
  std::string motionName;
  YAML::Node motionNode;

  for ( size_t i = 0; i < numMotion; ++i ) {
    get_required( meshMotionNode[i], "name", motionName );
    if ( motionName == theStringParams[0] ) {
      motionNode = meshMotionNode[i];
      break;
    }
  }

  if( motionName.empty() )
    throw std::runtime_error("WindEnergyAuxFunction::error() Can not find mesh motion name " + theStringParams[0]);

  // extract omega, unit vector, and centroid
  double mmOmega;
  get_required(motionNode, "omega", mmOmega);
  std::vector<double> unitVec;
  get_required(motionNode, "unit_vector", unitVec);

  // check if centroid needs to be computed
  std::vector<double> centroid(3,0.0);
  if ( motionNode["compute_centroids"] ) {
    std::vector<std::string> partNames = motionNode["mesh_parts"].as<std::vector<std::string>>();
    realm.compute_centroid_on_parts( partNames, centroid );
  }
  else {
    get_required(motionNode, "centroid_coordinates", centroid);
  }

  // fill member variables
  for ( size_t i = 0; i < 3; ++i ) {
    omegaMM_[i] = mmOmega*unitVec[i];
    centroidMM_[i] = centroid[i];
  }
}

WindEnergyAuxFunction::~WindEnergyAuxFunction()
{
  if ( NULL != tanhFunction_ )
    delete tanhFunction_;
}

void
WindEnergyAuxFunction::setup(const double time)
{
  if ( NULL != tanhFunction_  ) {
    omegaBlend_ = tanhFunction_->execute(time);
  }
}

void
WindEnergyAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  double c[3] = {0.0,0.0,0.0};
  double u[3] = {0.0,0.0,0.0};
  for(unsigned p=0; p < numPoints; ++p) {

    // define distance from centroid and compute cross product
    for ( unsigned i = 0; i < fieldSize; ++i )
      c[i] = coords[i] - centroidMM_[i];    
    cross_product(c, u);
    
    // allow for tanh blending
    for ( unsigned i = 0; i < fieldSize; ++i )
      fieldPtr[i] = u[i]*omegaBlend_;
  
    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

void
WindEnergyAuxFunction::cross_product(double *c, double *u) const
{
  u[0] = omegaMM_[1]*c[2] - omegaMM_[2]*c[1];
  u[1] = omegaMM_[2]*c[0] - omegaMM_[0]*c[2];
  u[2] = omegaMM_[0]*c[1] - omegaMM_[1]*c[0];
}

} // namespace nalu
} // namespace Sierra
