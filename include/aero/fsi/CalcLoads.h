// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CalcLoads_h
#define CalcLoads_h

#include <FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

class Realm;

class CalcLoads
{
public:
  CalcLoads(stk::mesh::PartVector& partVec, bool useShifted = true);
  ~CalcLoads();

  void setup(std::shared_ptr<stk::mesh::BulkData> bulk);

  void execute();

  //! Part vector over all wall boundary parts applying loads
  stk::mesh::PartVector partVec_;

  const bool useShifted_;

  std::shared_ptr<stk::mesh::BulkData> bulk_;

  VectorFieldType* coordinates_;
  ScalarFieldType* pressure_;
  ScalarFieldType* density_;
  ScalarFieldType* viscosity_;
  TensorFieldType* dudx_;
  GenericFieldType* exposedAreaVec_;
  GenericFieldType* tforceSCS_;
};

} // namespace nalu
} // namespace sierra

#endif
