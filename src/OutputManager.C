// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <OutputManager.h>
#include <NaluParsing.h>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_util/environment/WallTime.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <string>
#include <sstream>
#include <set>

namespace sierra {
namespace nalu {
void
OutputManager::load(const YAML::Node& node)
{
  std::vector<const YAML::Node*> foundNodes;
  NaluParsingHelper::find_nodes_given_key("output", node, foundNodes);

  const int nFound = foundNodes.size();
  infoVec_.resize(nFound);
  indexVec_.resize(nFound);

  for (int i = 0; i < nFound; i++) {
    infoVec_[i].load(*foundNodes[i]);
  }

  // check for conflicts and unsupported behavior between multiple output
  // instances
  std::stringstream errors;
  int tempIoGroupValue = 0;
  for (int i = 0; i < nFound; i++) {
    if (infoVec_[i].hasCatalystOutput_) {
      if (hasCatalystOutput_)
        errors << "Repeated catalyst output section. Only 1 output can have "
                  "catalyst.";
      hasCatalystOutput_ = true;
      catalystInfoId_ = i;
    }
    if (infoVec_[i].hasRestartBlock_) {
      if (hasRestartBlock_)
        errors << "Repeated restart output section. Only 1 output can have "
                  "restart.";
      hasRestartBlock_ = true;
      restartInfoId_ = i;
    }
    if (infoVec_[i].serializedIOGroupSize_) {
      if (i == 0) {
        tempIoGroupValue = infoVec_[i].serializedIOGroupSize_;
      }
      if (tempIoGroupValue == infoVec_[i].serializedIOGroupSize_) {
        serializedIOGroupSize_ = tempIoGroupValue;
      } else {
        errors << "Serialized IO Group Size must be the same for all outputs";
      }
    }
  }

  if (!errors.str().empty()) {
    throw std::runtime_error(errors.str());
  }
}

void
OutputManager::create_output_mesh(stk::io::StkMeshIoBroker* ioBroker)
{
  const auto& metaData = ioBroker->meta_data();
  const int nOutputs = infoVec_.size();
  for (int i = 0; i < nOutputs; i++) {
    OutputInfo* outputInfo = &(infoVec_[i]);
    std::string oname = outputInfo->outputDBName_;
    if (
      !outputInfo->catalystFileName_.empty() ||
      !outputInfo->paraviewScriptName_.empty()) {
#ifdef NALU_USES_CATALYST
      outputInfo->outputPropertyManager_->add(Ioss::Property(
        "CATALYST_BLOCK_PARSE_JSON_STRING", outputInfo->catalystParseJson_));
      std::string input_deck_name = "%B";
      stk::util::filename_substitution(input_deck_name);
      outputInfo->outputPropertyManager_->add(Ioss::Property(
        "CATALYST_BLOCK_PARSE_INPUT_DECK_NAME", input_deck_name));

      if (!outputInfo->paraviewScriptName_.empty())
        outputInfo->outputPropertyManager_->add(Ioss::Property(
          "CATALYST_SCRIPT", outputInfo->paraviewScriptName_.c_str()));

      outputInfo->outputPropertyManager_->add(
        Ioss::Property("CATALYST_CREATE_SIDE_SETS", 1));

      indexVec_[i] = ioBroker->create_output_mesh(
        oname, stk::io::WRITE_RESULTS, *outputInfo->outputPropertyManager_,
        "catalyst");
#else
      throw std::runtime_error("Nalu-Wind not built with Catalyst support");
#endif
    } else {
      indexVec_[i] = ioBroker->create_output_mesh(
        oname, stk::io::WRITE_RESULTS, *outputInfo->outputPropertyManager_);
      // create selector and limit output if necessary
      if (!outputInfo->targetNames_.empty()) {
        // extract part
        auto& partNameList = outputInfo->targetNames_;
        stk::mesh::PartVector searchParts;
        for (size_t k = 0; k < partNameList.size(); ++k) {
          stk::mesh::Part* thePart = metaData.get_part(partNameList[k]);
          if (NULL != thePart)
            searchParts.push_back(thePart);
          else
            throw std::runtime_error(
              "OutputManager::create_output_mesh: Part is null" +
              partNameList[k]);
        }

        // selector and bucket loop
        stk::mesh::Selector s_locally_owned =
          metaData.locally_owned_part() & stk::mesh::selectUnion(searchParts);
        // restrict output to the selector and entity rank
        ioBroker->set_output_selector(
          indexVec_[i], get_entity_rank(outputInfo, metaData), s_locally_owned);
      }
    }

    // Tell stk_io how to output element block nodal fields:
    // if 'true' passed to function, then output them as nodeset fields;
    // if 'false', then output as nodal fields (on all nodes of the mesh,
    // zero-filled) The option is provided since some
    // post-processing/visualization codes do not correctly handle nodeset
    // fields.
    ioBroker->use_nodeset_for_part_nodes_fields(
      indexVec_[i], outputInfo->outputNodeSet_);

    // FIXME: add_field can take user-defined output name, not just varName
    for (std::set<std::string>::iterator itorSet =
           outputInfo->outputFieldNameSet_.begin();
         itorSet != outputInfo->outputFieldNameSet_.end(); ++itorSet) {
      std::string varName = *itorSet;
      stk::mesh::FieldBase* theField =
        stk::mesh::get_field_by_name(varName, metaData);
      if (NULL == theField) {
        NaluEnv::self().naluOutputP0()
          << " Sorry, no field by the name " << varName << std::endl;
      } else {
        // 'varName' is the name that will be written to the database
        // For now, just using the name of the stk field
        ioBroker->add_field(indexVec_[i], *theField, varName);
      }
    }
  }
}

void
OutputManager::perform_outputs(
  const int timeStepCount,
  const double currentTime,
  stk::io::StkMeshIoBroker* ioBroker,
  const double wallTimeStart)
{
  const auto& metaData = ioBroker->meta_data();
  const int nOutputs = infoVec_.size();
  for (int i = 0; i < nOutputs; i++) {
    auto* outputInfo = &(infoVec_[i]);
    const auto resultsFileIndex = indexVec_[i];
    const int modStep = timeStepCount - outputInfo->outputStart_;
    const double elapsedWallTime = stk::wall_time() - wallTimeStart;
    // find the max over all core
    double g_elapsedWallTime = 0.0;
    stk::all_reduce_max(
      NaluEnv::self().parallel_comm(), &elapsedWallTime, &g_elapsedWallTime, 1);
    // convert to hours
    g_elapsedWallTime /= 3600.0;

    // check for elapsed WALL time threshold
    bool forcedOutput = false;
    if (outputInfo->userWallTimeResults_.first) {
      // only force output the first time the timer is exceeded
      if (g_elapsedWallTime > outputInfo->userWallTimeResults_.second) {
        forcedOutput = true;
        outputInfo->userWallTimeResults_.first = false;
        NaluEnv::self().naluOutputP0()
          << "Realm::provide_output()::Forced Result output will be processed "
             "at current time: "
          << currentTime << std::endl;
        NaluEnv::self().naluOutputP0()
          << " Elapsed (max) WALL time: " << g_elapsedWallTime << " (hours)"
          << std::endl;
      }
    }

    const bool isOutput = (timeStepCount >= outputInfo->outputStart_ &&
                           modStep % outputInfo->outputFreq_ == 0) ||
                          forcedOutput;

    if (isOutput) { /*
       NaluEnv::self().naluOutputP0()
         << "Realm shall provide output files at : currentTime/timeStepCount: "
         << currentTime << "/" << timeStepCount << " (" << name_ << ")"
         << std::endl;*/

      // Sync fields to host on NGP builds before output
      for (auto* fld : metaData.get_fields()) {
        fld->sync_to_host();
      }

      ioBroker->process_output_request(resultsFileIndex, currentTime);
    }
  }
}

stk::mesh::EntityRank
OutputManager::get_entity_rank(
  const OutputInfo* oInfo, const stk::mesh::MetaData& metaData)
{
  if (oInfo->targetType_ == "siderank") {
    return metaData.side_rank();
  }
  // not sure if I should make an option for element rank as well?
  return stk::topology::NODE_RANK;
}

} // namespace nalu
} // namespace sierra