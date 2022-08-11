// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef IdealGasPropertyEvaluator_h
#define IdealGasPropertyEvaluator_h

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

class IdealGasTPropertyEvaluator : public PropertyEvaluator
{
public:
  IdealGasTPropertyEvaluator(
    double pRef,
    double universalR,
    std::vector<std::pair<double, double>> mwMassFracVec);
  virtual ~IdealGasTPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  const double pRef_;
  const double R_;
  double mw_;
};

class IdealGasTYkPropertyEvaluator : public PropertyEvaluator
{
public:
  IdealGasTYkPropertyEvaluator(
    double pRef,
    double universalR,
    std::vector<double> mwVec,
    stk::mesh::MetaData& metaData);
  virtual ~IdealGasTYkPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  double compute_mw(const double* yk);

  // reference quantities
  const double pRef_;
  const double R_;

  // field definition and extraction,
  GenericFieldType* massFraction_;

  // reference mw vector; size and declaration
  size_t mwVecSize_;
  std::vector<double> mwVec_;
};

class IdealGasTPPropertyEvaluator : public PropertyEvaluator
{
public:
  IdealGasTPPropertyEvaluator(
    double universalR,
    std::vector<std::pair<double, double>> mwMassFracVec,
    stk::mesh::MetaData& metaData);
  virtual ~IdealGasTPPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  // reference quantities
  const double R_;

  // molecular weight computed at construction time
  double mw_;

  // field definition and extraction,
  ScalarFieldType* pressure_;
};

class IdealGasYkPropertyEvaluator : public PropertyEvaluator
{
public:
  IdealGasYkPropertyEvaluator(
    double pRef,
    double tRef,
    double universalR,
    std::vector<double> mwVec,
    stk::mesh::MetaData& metaData);
  virtual ~IdealGasYkPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  double compute_mw(const double* yk);

  // reference quantities
  const double pRef_;
  const double tRef_;
  const double R_;

  // field definition and extraction,
  GenericFieldType* massFraction_;

  // reference mw vector; size and declaration
  size_t mwVecSize_;
  std::vector<double> mwVec_;
};

} // namespace nalu
} // namespace sierra

#endif
