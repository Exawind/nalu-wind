// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef SteadyTaylorVortexContinuitySrcElemSuppAlg_h
#define SteadyTaylorVortexContinuitySrcElemSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;
class MasterElement;

class SteadyTaylorVortexContinuitySrcElemSuppAlg : public SupplementalAlgorithm
{
public:

  SteadyTaylorVortexContinuitySrcElemSuppAlg(
    Realm &realm);

  virtual ~SteadyTaylorVortexContinuitySrcElemSuppAlg() {}

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

  VectorFieldType *coordinates_;

  double dt_;
  const int nDim_;
  const double rhoP_;
  const double rhoS_;
  const double unot_;
  const double vnot_;
  const double znot_;
  const double pnot_;
  const double a_;
  const double amf_;
  const double Sc_;  
  const double pi_;
  double projTimeScale_;

  const bool useShifted_;

  // scratch space (at constructor)
  std::vector<double> scvCoords_;
  // at elem_resize
  std::vector<double> ws_shape_function_;
  std::vector<double> ws_coordinates_;
  std::vector<double> ws_scv_volume_;
};

} // namespace nalu
} // namespace Sierra

#endif
