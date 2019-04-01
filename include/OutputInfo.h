/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef OutputInfo_h
#define OutputInfo_h

#include <NaluParsing.h>

#include <string>
#include <set>

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
  //! List of fields explicity disabled by user for output
  std::set<std::string> disabledOutputSet_;
  std::set<std::string> restartFieldNameSet_;

};

} // namespace nalu
} // namespace Sierra

#endif
