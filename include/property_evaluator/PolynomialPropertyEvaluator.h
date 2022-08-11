// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef PolynomialPropertyEvaluator_h
#define PolynomialPropertyEvaluator_h

#include <property_evaluator/PropertyEvaluator.h>
#include <FieldTypeDef.h>

#include <string>
#include <map>
#include <vector>

namespace stk {
namespace mesh {
struct Entity;
}
} // namespace stk

namespace sierra {
namespace nalu {

class ReferencePropertyData;

class PolynomialPropertyEvaluator : public PropertyEvaluator
{
public:
  PolynomialPropertyEvaluator(
    const std::map<std::string, ReferencePropertyData*>&
      referencePropertyDataMap,
    const std::map<std::string, std::vector<double>>& lowPolynomialCoeffsMap,
    const std::map<std::string, std::vector<double>>& highPolynomialCoeffsMap,
    double universalR);
  virtual ~PolynomialPropertyEvaluator();

  virtual double execute(double* indVarList, stk::mesh::Entity node) = 0;

  const double universalR_;
  const size_t ykVecSize_;
  const double TlowHigh_;

  std::vector<double> mw_;
  std::vector<std::vector<double>> lowPolynomialCoeffs_;
  std::vector<std::vector<double>> highPolynomialCoeffs_;
};

} // namespace nalu
} // namespace sierra

#endif
