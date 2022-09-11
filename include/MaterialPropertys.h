// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MaterialPropertys_h
#define MaterialPropertys_h

#include <Enums.h>

#include <map>
#include <string>
#include <vector>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

class Realm;
class MaterialProperty;
class MaterialPropertyData;
class ReferencePropertyData;
class PropertyEvaluator;
class Simulation;

typedef std::vector<MaterialProperty*> MaterialPropertyVector;

class MaterialPropertys
{
public:
  MaterialPropertys(Realm& realm);

  ~MaterialPropertys();

  void load(const YAML::Node& node);

  void breadboard(){};

  // ease of access methods to particular initial condition
  size_t size() { return materialPropertyVector_.size(); }
  MaterialProperty* operator[](int i) { return materialPropertyVector_[i]; }

  Realm& realm_;
  MaterialPropertyVector materialPropertyVector_;
  std::string propertyTableName_;

  // vectors and maps required to manage full set of options
  std::vector<std::string> targetNames_;
  std::map<std::string, double> universalConstantMap_;
  std::map<PropertyIdentifier, MaterialPropertyData*> propertyDataMap_;
  std::map<std::string, ReferencePropertyData*>
    referencePropertyDataMap_; /* defines overall species ordering */
  std::map<PropertyIdentifier, PropertyEvaluator*> propertyEvalMap_;
  std::map<std::string, ReferencePropertyData*> tablePropertyMap_;
};

} // namespace nalu
} // namespace sierra

#endif
