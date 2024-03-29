// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef InverseDualVolumePropAlgorithm_h
#define InverseDualVolumePropAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

namespace stk {
namespace mesh {
class FieldBase;
class Part;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class Realm;

class InverseDualVolumePropAlgorithm : public Algorithm
{
public:
  InverseDualVolumePropAlgorithm(
    Realm& realm, stk::mesh::Part* part, stk::mesh::FieldBase* prop);

  virtual ~InverseDualVolumePropAlgorithm();

  virtual void execute();

  stk::mesh::FieldBase* prop_;
  ScalarFieldType* dualNodalVolume_;
};

} // namespace nalu
} // namespace sierra

#endif
