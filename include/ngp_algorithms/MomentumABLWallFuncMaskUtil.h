// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMABLWALLFUNCMASKUTIL_H_
#define MOMENTUMABLWALLFUNCMASKUTIL_H_

#include <Algorithm.h>
#include <stk_mesh/base/Types.hpp>

namespace sierra{
namespace nalu{

class Realm;

class MomentumABLWallFuncMaskUtil : public Algorithm{
public:
  MomentumABLWallFuncMaskUtil(Realm&, stk::mesh::Part*);
  virtual ~MomentumABLWallFuncMaskUtil() = default;
  void execute() override;
private:
  unsigned maskNodeIndex_{stk::mesh::InvalidOrdinal};

};

}
}
#endif /* MOMENTUMABLWALLFUNCMASKUTIL_H_ */
