// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SutherlandsPropertyEvaluator_h
#define SutherlandsPropertyEvaluator_h

#include <property_evaluator/PropertyEvaluator.h>
#include <FieldTypeDef.h>

#include <vector>
#include <map>

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

class ReferencePropertyData;

class SutherlandsPropertyEvaluator : public PropertyEvaluator
{
public:
  SutherlandsPropertyEvaluator(
    const std::map<std::string, ReferencePropertyData*>&
      referencePropertyDataMap,
    const std::map<std::string, std::vector<double>>& polynomialCoeffsMap);
  virtual ~SutherlandsPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  double compute_viscosity(const double& T, const double* pt_poly);

  std::vector<double> refMassFraction_;
  std::vector<std::vector<double>> polynomialCoeffs_;
};

class SutherlandsYkPropertyEvaluator : public PropertyEvaluator
{
public:
  SutherlandsYkPropertyEvaluator(
    const std::map<std::string, std::vector<double>>& polynomialCoeffsMap,
    stk::mesh::MetaData& metaData);

  virtual ~SutherlandsYkPropertyEvaluator();

  virtual double execute(double* indVarList, stk::mesh::Entity node);

  virtual double compute_viscosity(const double& T, const double* pt_poly);

  // field definition and extraction
  GenericFieldType* massFraction_;
  size_t ykVecSize_;

  // polynomial coeffs
  std::vector<std::vector<double>> polynomialCoeffs_;
};

class SutherlandsYkTrefPropertyEvaluator : public SutherlandsYkPropertyEvaluator
{
public:
  SutherlandsYkTrefPropertyEvaluator(
    const std::map<std::string, std::vector<double>>& polynomialCoeffsMap,
    stk::mesh::MetaData& metaData,
    const double tRef);

  virtual ~SutherlandsYkTrefPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  const double tRef_;
};

} // namespace nalu
} // namespace sierra

#endif
