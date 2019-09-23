/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <CoriolisSrc.h>
#include <Realm.h>
#include <SolutionOptions.h>

#include <master_element/TensorOps.h>

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// CoriolisSrc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
CoriolisSrc::CoriolisSrc(const SolutionOptions &solnOpts)
{
  pi_ = std::acos(-1.0);

  // extract user parameters from solution options
  earthAngularVelocity_ = solnOpts.earthAngularVelocity_;
  latitude_ = solnOpts.latitude_ * pi_ / 180.0;

  ThrowRequire(solnOpts.eastVector_.size() == nDim_);
  ThrowRequire(solnOpts.northVector_.size() == nDim_);

  for (int i=0; i < nDim_; ++i) {
    eastVector_[i] = solnOpts.eastVector_[i];
    northVector_[i] = solnOpts.northVector_[i];
  }

  // normalize the east and north vectors
  double magE = std::sqrt(eastVector_[0]*eastVector_[0]+eastVector_[1]*eastVector_[1]+eastVector_[2]*eastVector_[2]);
  double magN = std::sqrt(northVector_[0]*northVector_[0]+northVector_[1]*northVector_[1]+northVector_[2]*northVector_[2]);
  for (int i=0; i<nDim_; ++i) {
    eastVector_[i] /= magE;
    northVector_[i] /= magN;
  }

  // calculate the 'up' unit vector
  cross3(eastVector_, northVector_, upVector_);

  // some factors that do not change
  sinphi_ = std::sin(latitude_);
  cosphi_ = std::cos(latitude_);
  corfac_ = 2.0*earthAngularVelocity_;

  // Jacobian entries
  Jxy_ = corfac_ * ((eastVector_[0]*northVector_[1]-northVector_[0]*eastVector_[1])*sinphi_
                 + (upVector_[0]*eastVector_[1]-eastVector_[0]*upVector_[1])*cosphi_);
  Jxz_ = corfac_ * ((eastVector_[0]*northVector_[2]-northVector_[0]*eastVector_[2])*sinphi_
                 + (upVector_[0]*eastVector_[2]-eastVector_[0]*upVector_[2])*cosphi_);
  Jyz_ = corfac_ * ((eastVector_[1]*northVector_[2]-northVector_[1]*eastVector_[2])*sinphi_
                 + (upVector_[1]*eastVector_[2]-eastVector_[1]*upVector_[2])*cosphi_);
}

} // namespace nalu
} // namespace Sierra
