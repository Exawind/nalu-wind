// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef MomentumNSOKeElemSuppAlgDep_h
#define MomentumNSOKeElemSuppAlgDep_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;
class MasterElement;

class MomentumNSOKeElemSuppAlgDep : public SupplementalAlgorithm
{
public:

  MomentumNSOKeElemSuppAlgDep(
    Realm &realm,
    VectorFieldType *velocity,
    GenericFieldType *Gju,
    const double fourthFac);

  virtual ~MomentumNSOKeElemSuppAlgDep() {}

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

  VectorFieldType *velocityNp1_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *pressure_;
  VectorFieldType *velocityRTM_;
  VectorFieldType *coordinates_;
  GenericFieldType *Gju_;
  VectorFieldType *Gjp_;
  const int nDim_;
  const double Cupw_;
  const double small_;
  const double fourthFac_;
  const bool useShiftedGradOp_;

  // fixed space
  std::vector<double> ws_rhoVrtmScs_;
  std::vector<double> ws_uNp1Scs_;
  std::vector<double> ws_dpdxScs_;
  std::vector<double> ws_GjpScs_;
  std::vector<double> ws_dkedxScs_;
  std::vector<double> ws_kd_;

  // scratch space; geometry
  std::vector<double> ws_scs_areav_;
  std::vector<double> ws_dndx_;
  std::vector<double> ws_deriv_;
  std::vector<double> ws_det_j_;
  std::vector<double> ws_shape_function_;
  std::vector<double> ws_gUpper_;
  std::vector<double> ws_gLower_;

  // scratch space; fields
  std::vector<double> ws_velocityNp1_;
  std::vector<double> ws_rhoNp1_;
  std::vector<double> ws_pressure_;
  std::vector<double> ws_velocityRTM_;
  std::vector<double> ws_coordinates_;
  std::vector<double> ws_Gju_;
  std::vector<double> ws_Gjp_;
  std::vector<double> ws_ke_;
};

} // namespace nalu
} // namespace Sierra

#endif
