// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef WaterPropertyEvaluator_h
#define WaterPropertyEvaluator_h

#include <property_evaluator/PropertyEvaluator.h>
#include <FieldTypeDef.h>

#include <vector>

namespace stk {
namespace mesh {
struct Entity;
}
} // namespace stk

namespace stk {
namespace mesh {
class MetaData;
}
} // namespace stk

namespace sierra {
namespace nalu {

class WaterDensityTPropertyEvaluator : public PropertyEvaluator
{
public:
  WaterDensityTPropertyEvaluator(stk::mesh::MetaData& metaData);
  virtual ~WaterDensityTPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
};

class WaterViscosityTPropertyEvaluator : public PropertyEvaluator
{
public:
  WaterViscosityTPropertyEvaluator(stk::mesh::MetaData& metaData);
  virtual ~WaterViscosityTPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
  const double dw_;
};

class WaterSpecHeatTPropertyEvaluator : public PropertyEvaluator
{
public:
  WaterSpecHeatTPropertyEvaluator(stk::mesh::MetaData& metaData);
  virtual ~WaterSpecHeatTPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
  const double dw_;
  const double ew_;
};

class WaterEnthalpyTPropertyEvaluator : public PropertyEvaluator
{
public:
  WaterEnthalpyTPropertyEvaluator(stk::mesh::MetaData& metaData);
  virtual ~WaterEnthalpyTPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  double compute_h(const double T);

  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
  const double dw_;
  const double ew_;
  const double Tref_;
  const double hRef_;
};

class WaterThermalCondTPropertyEvaluator : public PropertyEvaluator
{
public:
  WaterThermalCondTPropertyEvaluator(stk::mesh::MetaData& metaData);
  virtual ~WaterThermalCondTPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
};

} // namespace nalu
} // namespace sierra

#endif
