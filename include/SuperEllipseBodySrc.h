// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SuperEllipseBodySrc_h
#define SuperEllipseBodySrc_h

#include "KokkosInterface.h"
#include "vs/vector_space.h"

namespace sierra {
namespace nalu {

class SolutionOptions;

class SuperEllipseBodySrc
{
public:
  SuperEllipseBodySrc(const SolutionOptions& solnOpts);

  SuperEllipseBodySrc(const SolutionOptions& solnOpts, vs::Vector loc, 
                      vs::Vector orient, vs::Vector dim);

  KOKKOS_DEFAULTED_FUNCTION
  SuperEllipseBodySrc() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~SuperEllipseBodySrc() = default;

  void read_from_file();

  vs::Vector get_loc() const { return seb_loc_; }
  vs::Vector get_orient() const { return seb_orient_; }
  vs::Vector get_dim() const { return seb_dim_; }

private:

  static constexpr int nDim_ = 3;

  std::string seb_file_;
  
  vs::Vector seb_loc_;
  vs::Vector seb_orient_;
  vs::Vector seb_dim_;

};

} // namespace nalu
} // namespace sierra

#endif
