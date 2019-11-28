// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ScalarNSOElemSuppAlgDep_h
#define ScalarNSOElemSuppAlgDep_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;
class MasterElement;

class ScalarNSOElemSuppAlgDep : public SupplementalAlgorithm
{
public:

  ScalarNSOElemSuppAlgDep(
    Realm &realm,
    ScalarFieldType *scalarQ,
    VectorFieldType *Gjq,
    ScalarFieldType *diffFluxCoeff,
    const double fourthFac,
    const double altResFac);

  virtual ~ScalarNSOElemSuppAlgDep() {}

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

  ScalarFieldType *scalarQNm1_;
  ScalarFieldType *scalarQN_;
  ScalarFieldType *scalarQNp1_;
  ScalarFieldType *densityNm1_;
  ScalarFieldType *densityN_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *diffFluxCoeff_;
  VectorFieldType *velocityRTM_;
  VectorFieldType *Gjq_;
  VectorFieldType *coordinates_;

  double dt_;
  const int nDim_;
  double gamma1_;
  double gamma2_;
  double gamma3_;
  const double Cupw_;
  const double small_;
  const double fourthFac_;
  const double altResFac_;
  const double om_altResFac_;
  const double nonConservedForm_;
  const bool useShiftedGradOp_;

  // fixed space
  std::vector<double> ws_dqdxScs_;
  std::vector<double> ws_rhoVrtmScs_;

  // scratch space; geometry
  std::vector<double> ws_scs_areav_;
  std::vector<double> ws_dndx_;
  std::vector<double> ws_deriv_;
  std::vector<double> ws_det_j_;
  std::vector<double> ws_shape_function_;
  std::vector<double> ws_gUpper_;
  std::vector<double> ws_gLower_;

  // scratch space; fields
  std::vector<double> ws_qNm1_;
  std::vector<double> ws_qN_;
  std::vector<double> ws_qNp1_;
  std::vector<double> ws_rhoNm1_;
  std::vector<double> ws_rhoN_;
  std::vector<double> ws_rhoNp1_;
  std::vector<double> ws_velocityRTM_;
  std::vector<double> ws_diffFluxCoeff_;
  std::vector<double> ws_Gjq_;
  std::vector<double> ws_coordinates_;
};

} // namespace nalu
} // namespace Sierra

#endif
