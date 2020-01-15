// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef GenericPropAlgorithm_h
#define GenericPropAlgorithm_h

#include <Algorithm.h>

namespace stk {
namespace mesh {
class FieldBase;
class Part;
}
}

namespace sierra{
namespace nalu{

class Realm;
class PropertyEvaluator;

class GenericPropAlgorithm : public Algorithm
{
public:

  GenericPropAlgorithm(
    Realm & realm,
    stk::mesh::Part * part,
    stk::mesh::FieldBase * prop,
    PropertyEvaluator *propEvaluator);

  virtual ~GenericPropAlgorithm() {}

  virtual void execute();

  stk::mesh::FieldBase *prop_;
  PropertyEvaluator *propEvaluator_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
