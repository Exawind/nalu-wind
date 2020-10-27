// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MISCIDDESABL_H_
#define MISCIDDESABL_H_

namespace stk{
namespace mesh{
  class MetaData;
  class Part;
}
}

namespace sierra {
namespace nalu{

class Realm;

void register_iddes_abl_fields(stk::mesh::MetaData& meta_data, stk::mesh::Part* part);
void initial_work_iddes_abl(Realm& realm);

}
}
#endif /* MISCIDDESABL_H_ */
