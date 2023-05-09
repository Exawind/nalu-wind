// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CopyFieldAlgorithm_h
#define CopyFieldAlgorithm_h

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

class CopyFieldAlgorithm : public Algorithm
{
public:
  CopyFieldAlgorithm(
    Realm& realm,
    const stk::mesh::PartVector& part_vec,
    stk::mesh::FieldBase* fromField,
    stk::mesh::FieldBase* toField,
    const unsigned beginPos,
    const unsigned endPos,
    const stk::mesh::EntityRank entityRank);

  CopyFieldAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    stk::mesh::FieldBase* fromField,
    stk::mesh::FieldBase* toField,
    const unsigned beginPos,
    const unsigned endPos,
    const stk::mesh::EntityRank entityRank);

  virtual ~CopyFieldAlgorithm() {}
  virtual void execute();

private:
  stk::mesh::FieldBase* fromField_;
  stk::mesh::FieldBase* toField_;

  const unsigned beginPos_;
  const unsigned endPos_;
  const stk::mesh::EntityRank entityRank_;

private:
  // make this non-copyable
  CopyFieldAlgorithm(const CopyFieldAlgorithm& other);
  CopyFieldAlgorithm& operator=(const CopyFieldAlgorithm& other);
};

} // namespace nalu
} // namespace sierra

#endif
