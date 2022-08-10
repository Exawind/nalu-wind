// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TemperaturePropAlgorithm_h
#define TemperaturePropAlgorithm_h

#include <Algorithm.h>

// standard c++
#include <string>

namespace stk {
namespace mesh {
class FieldBase;
class Part;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class Realm;
class PropertyEvaluator;

class TemperaturePropAlgorithm : public Algorithm
{
public:
  TemperaturePropAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    stk::mesh::FieldBase* prop,
    PropertyEvaluator* propEvaluator,
    std::string tempName = "temperature");

  virtual ~TemperaturePropAlgorithm() {}

  virtual void execute();

  stk::mesh::FieldBase* prop_;
  PropertyEvaluator* propEvaluator_;
  stk::mesh::FieldBase* temperature_;
};

} // namespace nalu
} // namespace sierra

#endif
