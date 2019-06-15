/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef CoriolisSrc_h
#define CoriolisSrc_h

#include "KokkosInterface.h"

namespace sierra{
namespace nalu{

class SolutionOptions;

class CoriolisSrc {
public:
  CoriolisSrc(const SolutionOptions& solnOpts);

  KOKKOS_FUNCTION
  CoriolisSrc() = default;

  KOKKOS_FUNCTION
  virtual ~CoriolisSrc() = default;

  static constexpr int nDim_ = 3;

  double eastVector_[nDim_];
  double northVector_[nDim_];
  double upVector_[nDim_];

  double earthAngularVelocity_;
  double latitude_;
  double sinphi_;
  double cosphi_;
  double corfac_;
  double Jxy_, Jxz_, Jyz_;
  double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif
