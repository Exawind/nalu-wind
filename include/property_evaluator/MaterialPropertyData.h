// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MaterialPropertyData_h
#define MaterialPropertyData_h

#include <Enums.h>

#include <string>
#include <vector>
#include <map>

namespace sierra {
namespace nalu {

class MaterialPropertyData
{
public:
  MaterialPropertyData();
  ~MaterialPropertyData();

  MaterialPropertyType type_;
  double constValue_;

  // mixture fraction specifics
  double primary_;
  double secondary_;

  // table specifics, all single in size, all possibly required to be more
  // general
  std::vector<std::string> indVarName_;
  std::vector<std::string> indVarTableName_;
  std::string auxVarName_;
  std::string tablePropName_;
  std::string tableAuxVarName_;

  // generic property name
  std::string genericPropertyEvaluatorName_;

  // vectors and maps
  std::map<std::string, std::vector<double>> polynomialCoeffsMap_;
  std::map<std::string, std::vector<double>> lowPolynomialCoeffsMap_;
  std::map<std::string, std::vector<double>> highPolynomialCoeffsMap_;
  std::map<std::string, double> cpConstMap_;
  std::map<std::string, double> hfConstMap_;
};

} // namespace nalu
} // namespace sierra

#endif
