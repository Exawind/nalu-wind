// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SpecificHeatPropertyEvaluator_h
#define SpecificHeatPropertyEvaluator_h

#include <property_evaluator/PolynomialPropertyEvaluator.h>
#include <FieldTypeDef.h>

#include <string>
#include <map>
#include <vector>

namespace stk {
namespace mesh {
struct Entity;
class MetaData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class ReferencePropertyData;

class SpecificHeatPropertyEvaluator : public PolynomialPropertyEvaluator
{
public:
  SpecificHeatPropertyEvaluator(
    const std::map<std::string, ReferencePropertyData*>&
      referencePropertyDataMap,
    const std::map<std::string, std::vector<double>>& lowPolynomialCoeffsMap,
    const std::map<std::string, std::vector<double>>& highPolynomialCoeffsMap,
    double universalR);
  virtual ~SpecificHeatPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  double compute_cp_r(const double& T, const double* pt_poly);

  std::vector<double> refMassFraction_;
};

class SpecificHeatTYkPropertyEvaluator : public PolynomialPropertyEvaluator
{
public:
  SpecificHeatTYkPropertyEvaluator(
    const std::map<std::string, ReferencePropertyData*>&
      referencePropertyDataMap,
    const std::map<std::string, std::vector<double>>& lowPolynomialCoeffsMap,
    const std::map<std::string, std::vector<double>>& highPolynomialCoeffsMap,
    double universalR,
    stk::mesh::MetaData& metaData);

  virtual ~SpecificHeatTYkPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  double compute_cp_r(const double& T, const double* pt_poly);

  // field definition and extraction
  GenericFieldType* massFraction_;
};

class SpecificHeatConstCpkPropertyEvaluator : public PropertyEvaluator
{
public:
  SpecificHeatConstCpkPropertyEvaluator(
    const std::map<std::string, double>& cpConstMap,
    stk::mesh::MetaData& metaData);

  virtual ~SpecificHeatConstCpkPropertyEvaluator();

  double execute(double* indVarList, stk::mesh::Entity node);

  // field definition and extraction
  const size_t cpVecSize_;
  GenericFieldType* massFraction_;
  std::vector<double> cpVec_;
};

} // namespace nalu
} // namespace sierra

#endif
