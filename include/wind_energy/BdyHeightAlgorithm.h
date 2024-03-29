// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef BDYHEIGHTALGORITHM_H
#define BDYHEIGHTALGORITHM_H

#include "FieldTypeDef.h"

#include <vector>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

class Realm;

class BdyHeightAlgorithm
{
public:
  BdyHeightAlgorithm(Realm& realm) : realm_(realm) {}

  virtual ~BdyHeightAlgorithm() {}

  virtual void calc_height_levels(
    stk::mesh::Selector&, ScalarIntFieldType&, std::vector<double>&) = 0;

protected:
  Realm& realm_;

private:
  BdyHeightAlgorithm() = delete;
  BdyHeightAlgorithm(const BdyHeightAlgorithm&) = delete;
};

class RectilinearMeshHeightAlg : public BdyHeightAlgorithm
{
public:
  RectilinearMeshHeightAlg(Realm&, const YAML::Node&);

  virtual ~RectilinearMeshHeightAlg() {}

  /** Determine the unique height levels in this mesh
   */
  virtual void calc_height_levels(
    stk::mesh::Selector&, ScalarIntFieldType&, std::vector<double>&) override;

protected:
  //! Process yaml inputs and initialize the class data
  void load(const YAML::Node&);

  //! Multiplier to convert doubles to int for unique heights mapping
  double heightMultiplier_{1.0e6};

  //! Mimum height to account for negative values in the wall normal direction
  double hMin_{0.0};

  /** Index of the wall normal direction
   *
   *  x = 1; y = 2, z = 3
   */
  int wallNormIndex_{3};

private:
  RectilinearMeshHeightAlg() = delete;
  RectilinearMeshHeightAlg(const RectilinearMeshHeightAlg&) = delete;
};

} // namespace nalu
} // namespace sierra

#endif /* BDYHEIGHTALGORITHM_H */
