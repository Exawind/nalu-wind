// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef StringTimeCoordFunction_h
#define StringTimeCoordFunction_h

#include "stk_expreval/Evaluator.hpp"

#include "Kokkos_Core.hpp"

#include <string>

namespace sierra::nalu {

namespace fcn {
inline constexpr int UNMAPPED_INDEX = -1;
}

class StringTimeCoordFunction
{
public:
  StringTimeCoordFunction(std::string fcn);
  KOKKOS_FUNCTION double
  operator()(double t, double x, double y, double z) const;
  [[nodiscard]] KOKKOS_FUNCTION bool is_constant() const { return constant; }

  KOKKOS_FUNCTION int spatial_dim() const
  {
    if (z_index != fcn::UNMAPPED_INDEX) {
      return 3;
    } else if (y_index != fcn::UNMAPPED_INDEX) {
      return 2;
    } else if (x_index != fcn::UNMAPPED_INDEX) {
      return 1;
    }
    return 0;
  }

  [[nodiscard]] KOKKOS_FUNCTION bool is_spatial() const
  {
    return spatial_dim() != 0;
  }

private:
  stk::expreval::ParsedEval<> parsed_eval;
  bool constant = false;
  int t_index = fcn::UNMAPPED_INDEX;
  int x_index = fcn::UNMAPPED_INDEX;
  int y_index = fcn::UNMAPPED_INDEX;
  int z_index = fcn::UNMAPPED_INDEX;
};

} // namespace sierra::nalu

#endif
