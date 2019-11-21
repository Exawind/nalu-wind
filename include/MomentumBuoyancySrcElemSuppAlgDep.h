// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef MomentumBuoyancySrcElemSuppAlgDep_h
#define MomentumBuoyancySrcElemSuppAlgDep_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;
class MasterElement;

class MomentumBuoyancySrcElemSuppAlgDep : public SupplementalAlgorithm
{
public:

  MomentumBuoyancySrcElemSuppAlgDep(
    Realm &realm);

  virtual ~MomentumBuoyancySrcElemSuppAlgDep() {}

  virtual void setup();

  virtual void elem_resize(
    MasterElement *meSCS,
    MasterElement *meSCV);

  virtual void elem_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity element,
    MasterElement *meSCS,
    MasterElement *meSCV);
  
  const stk::mesh::BulkData *bulkData_;

  ScalarFieldType *densityNp1_;
  VectorFieldType *coordinates_;

  const int nDim_;
  double rhoRef_;
  const bool useShifted_;
  std::vector<double> gravity_;

  // scratch space
  std::vector<double> ws_shape_function_;
  std::vector<double> ws_rhoNp1_;
  std::vector<double> ws_coordinates_;
  std::vector<double> ws_scv_volume_;
};

} // namespace nalu
} // namespace Sierra

#endif
