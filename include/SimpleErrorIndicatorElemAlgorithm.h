// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef SimpleErrorIndicatorElemAlgorithm_h
#define SimpleErrorIndicatorElemAlgorithm_h

#include<Algorithm.h>
#include<FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

class SimpleErrorIndicatorElemAlgorithm : public Algorithm
{
public:

  SimpleErrorIndicatorElemAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  ~SimpleErrorIndicatorElemAlgorithm();

  void execute();

  // extract fields; nodal
  VectorFieldType *velocity_;
  VectorFieldType *coordinates_;
  GenericFieldType *dudx_;
  GenericFieldType *errorIndicatorField_;

};

} // namespace nalu
} // namespace Sierra

#endif
