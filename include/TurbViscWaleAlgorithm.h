// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TurbViscWaleAlgorithm_h
#define TurbViscWaleAlgorithm_h

#include <Algorithm.h>

#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;

class TurbViscWaleAlgorithm : public Algorithm
{
public:
  TurbViscWaleAlgorithm(Realm& realm, stk::mesh::Part* part);
  virtual ~TurbViscWaleAlgorithm() {}
  virtual void execute();

  TensorFieldType* dudx_;
  ScalarFieldType* density_;
  ScalarFieldType* tvisc_;
  ScalarFieldType* dualNodalVolume_;

  const double Cw_;
  const double kappa_;
};

} // namespace nalu
} // namespace sierra

#endif
