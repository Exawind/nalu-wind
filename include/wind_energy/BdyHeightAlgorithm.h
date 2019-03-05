/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef BDYHEIGHTALGORITHM_H
#define BDYHEIGHTALGORITHM_H

#include "FieldTypeDef.h"
#include "NaluParsing.h"

#include <vector>

namespace sierra {
namespace nalu {

class Realm;

class BdyHeightAlgorithm
{
public:
  BdyHeightAlgorithm(Realm& realm) : realm_(realm) {}

  virtual ~BdyHeightAlgorithm() {}

  virtual void calc_height_levels(
    stk::mesh::Selector&,
    ScalarIntFieldType&,
    std::vector<double>&) = 0;

protected:
  Realm& realm_;

private:
  BdyHeightAlgorithm() = delete;
  BdyHeightAlgorithm(const BdyHeightAlgorithm&) = delete;
};

class RectilinearMeshHeightAlg : public BdyHeightAlgorithm
{
public:
  RectilinearMeshHeightAlg(
    Realm&,
    const YAML::Node&);

  virtual ~RectilinearMeshHeightAlg() {}

  /** Determine the unique height levels in this mesh
   */
  virtual void calc_height_levels(
    stk::mesh::Selector&,
    ScalarIntFieldType&,
    std::vector<double>&) override;

protected:
  //! Process yaml inputs and initialize the class data
  void load(const YAML::Node&);

  //! Multiplier to convert doubles to int for unique heights mapping
  double heightMultiplier_{1.0e3};

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

}  // nalu
}  // sierra


#endif /* BDYHEIGHTALGORITHM_H */
