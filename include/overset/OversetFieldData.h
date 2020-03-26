// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef OVERSETFIELDDATA_H
#define OVERSETFIELDDATA_H

namespace stk {
namespace mesh {
class FieldBase;
}
}

namespace sierra {
namespace nalu {

struct OversetFieldData
{
  OversetFieldData(stk::mesh::FieldBase* field, int sizeRow=1, int sizeCol=1)
    : field_(field),
      sizeRow_(sizeRow),
      sizeCol_(sizeCol)
  {}

  stk::mesh::FieldBase* field_;
  int sizeRow_;
  int sizeCol_;
};

}  // nalu
}  // sierra

#endif /* OVERSETFIELDDATA_H */
