// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef SupplementalAlgorithm_h
#define SupplementalAlgorithm_h

#include <master_element/MasterElement.h>
#include <KokkosInterface.h>
#include <vector>

#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class SupplementalAlgorithm
{
public:
  
  SupplementalAlgorithm(
    Realm &realm);
  
  virtual ~SupplementalAlgorithm() {}

  virtual void setup() {}

  virtual void elem_execute(
    double * /* lhs */,
    double * /* rhs */,
    stk::mesh::Entity  /* element */,
    MasterElement * /* meSCS */,
    MasterElement * /* meSCV */) {}
  
  virtual void node_execute(
    double * /* lhs */,
    double * /* rhs */,
    stk::mesh::Entity  /* node */) {}
  
  virtual void elem_resize(
    MasterElement * /* meSCS */,
    MasterElement * /* meSCV */) {}

  Realm &realm_;  
};

} // namespace nalu
} // namespace Sierra

#endif
