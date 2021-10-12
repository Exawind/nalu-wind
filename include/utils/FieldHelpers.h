// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#ifndef FIELDHELPERS_H_
#define FIELDHELPERS_H_

namespace stk {
namespace mesh {
class MetaData;
}
} // namespace stk

namespace sierra {
namespace nalu {
void populate_dnv_states(
  const stk::mesh::MetaData& meta,
  unsigned& nm1ID,
  unsigned& nID,
  unsigned& np1ID);
}
} // namespace sierra

#endif /* FIELDHELPERS_H_ */
