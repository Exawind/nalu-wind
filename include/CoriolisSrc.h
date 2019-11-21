// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



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
