// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AuxFunctionAlgorithm_h
#define AuxFunctionAlgorithm_h

#include <Algorithm.h>

#include <vector>
#include <stk_mesh/base/Types.hpp>

namespace stk {
namespace mesh {
class Part;
class FieldBase;
class Selector;

typedef std::vector<Part*> PartVector;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class AuxFunction;

class AuxFunctionAlgorithm : public Algorithm
{
public:
  AuxFunctionAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    stk::mesh::FieldBase* field,
    AuxFunction* auxFunction,
    stk::mesh::EntityRank entityRank);

  virtual ~AuxFunctionAlgorithm();
  virtual void execute();

private:
  stk::mesh::FieldBase* field_;
  AuxFunction* auxFunction_;
  stk::mesh::EntityRank entityRank_;

private:
  // make this non-copyable
  AuxFunctionAlgorithm(const AuxFunctionAlgorithm& other);
  AuxFunctionAlgorithm& operator=(const AuxFunctionAlgorithm& other);
};

} // namespace nalu
} // namespace sierra

#endif
