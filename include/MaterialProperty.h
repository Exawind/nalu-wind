// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef MaterialProperty_h
#define MaterialProperty_h

#include <Enums.h>

#include <map>
#include <string>
#include <vector>

namespace sierra{
namespace nalu{

class MaterialPropertys;

class MaterialProperty {
public:
  MaterialProperty(MaterialPropertys& matPropertys);
  
  ~MaterialProperty();
  
  void load(const YAML::Node & node);
  
  virtual void breadboard(){}

  Simulation *root();
  EquationSystems *parent();

  MaterialPropertys &matPropertys_;
};


} // namespace nalu
} // namespace Sierra

#endif
