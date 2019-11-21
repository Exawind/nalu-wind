// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef PostProcessingData_h
#define PostProcessingData_h

#include <string>
#include <vector>

namespace sierra{
namespace nalu{

class PostProcessingData {

 public:
  PostProcessingData() : type_("na"), physics_("na"), outputFileName_("na"), frequency_(10){}
  ~PostProcessingData() {}
  
  std::string type_;
  std::string physics_;
  std::string outputFileName_;
  int frequency_;
  std::vector<double> parameters_;
  std::vector<std::string> targetNames_;
};
 
} // namespace nalu
} // namespace Sierra

#endif
