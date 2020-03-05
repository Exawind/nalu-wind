// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef OutputInfo_h
#define OutputInfo_h

#include <string>
#include <set>

namespace YAML { class Node; }

namespace Ioss{
  class PropertyManager;
}

namespace sierra{
namespace nalu{

class OutputInfo
{
public:
  
  OutputInfo();
  ~OutputInfo();
  
  void load(const YAML::Node & node);

  int get_restart_frequency();
  
  // helper methods for compression options
  int get_output_compression();
  bool get_output_shuffle();
  
  int get_restart_compression();
  bool get_restart_shuffle();
  
  std::string outputDBName_;
  
  // catalyst options
  std::string catalystFileName_;
  std::string catalystParseJson_;
  std::string paraviewScriptName_;

  int outputFreq_;
  int outputStart_;
  bool outputNodeSet_; 
  int serializedIOGroupSize_;
  bool hasOutputBlock_;
  bool hasRestartBlock_;
  bool activateRestart_;
  bool meshAdapted_;
  double restartTime_;
  std::string restartDBName_;
  int restartFreq_;
  int restartStart_;
  int restartMaxDataBaseStepSize_;
  bool restartNodeSet_;
  int outputCompressionLevel_;
  bool outputCompressionShuffle_;
  int restartCompressionLevel_;
  bool restartCompressionShuffle_;

  std::pair<bool, double> userWallTimeResults_;
  std::pair<bool, double> userWallTimeRestart_;

  // manage the properties for io
  Ioss::PropertyManager *outputPropertyManager_;
  Ioss::PropertyManager *restartPropertyManager_;

  std::set<std::string> outputFieldNameSet_;
  std::set<std::string> restartFieldNameSet_;

};

} // namespace nalu
} // namespace Sierra

#endif
