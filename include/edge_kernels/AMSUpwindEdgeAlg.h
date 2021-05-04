// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AMSUPWINDEDGEALG_H_
#define AMSUPWINDEDGEALG_H_

#include <string>
#include <Algorithm.h>
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class AMSUpwindEdgeAlg : public Algorithm
{
public:
  using DblType = double;
  AMSUpwindEdgeAlg(Realm&, stk::mesh::Part*);
  virtual ~AMSUpwindEdgeAlg() = default;
  void execute() override;

private:
  const std::string velocityName_;
  unsigned pecletFactor_{stk::mesh::InvalidOrdinal};
  unsigned beta_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* AMSUPWINDEDGEALG_H_ */
