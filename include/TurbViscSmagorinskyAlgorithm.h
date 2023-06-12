// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TurbViscSmagorinskyAlgorithm_h
#define TurbViscSmagorinskyAlgorithm_h

#include <Algorithm.h>

#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;

class TurbViscSmagorinskyAlgorithm : public Algorithm
{
public:
  TurbViscSmagorinskyAlgorithm(Realm& realm, stk::mesh::Part* part);
  virtual ~TurbViscSmagorinskyAlgorithm() {}
  virtual void execute();

  TensorFieldType* dudx_;
  ScalarFieldType* density_;
  ScalarFieldType* tvisc_;
  ScalarFieldType* dualNodalVolume_;

  const double cmuCs_;
};

} // namespace nalu
} // namespace sierra

#endif
