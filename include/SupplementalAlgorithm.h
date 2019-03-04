/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


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
