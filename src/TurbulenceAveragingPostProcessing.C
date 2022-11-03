// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <TurbulenceAveragingPostProcessing.h>
#include <AveragingInfo.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <Realm.h>
#include <MovingAveragePostProcessor.h>
#include <SolutionOptions.h>

// NGP Algorithms
#include "ElemDataRequests.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpFieldManager.h"

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>

// basic c++
#include <stdexcept>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <complex>
#include <cmath>
#include <memory>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// TurbulenceAveragingPostProcessing - post process
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbulenceAveragingPostProcessing::TurbulenceAveragingPostProcessing(
  Realm& realm, const YAML::Node& node)
  : realm_(realm),
    currentTimeFilter_(0.0),
    timeFilterInterval_(1.0e8),
    forcedReset_(false)
{
  // load the data
  load(node);
}

TurbulenceAveragingPostProcessing::TurbulenceAveragingPostProcessing(
  Realm& realm)
  : realm_(realm),
    currentTimeFilter_(0.0),
    timeFilterInterval_(1.0e8),
    forcedReset_(false)
{
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TurbulenceAveragingPostProcessing::~TurbulenceAveragingPostProcessing()
{
  for (size_t k = 0; k < averageInfoVec_.size(); ++k)
    delete averageInfoVec_[k];
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::load(const YAML::Node& y_node)
{
  // output for results
  const YAML::Node y_average = y_node["turbulence_averaging"];
  if (y_average) {
    get_if_present(y_average, "forced_reset", forcedReset_, forcedReset_);
    get_if_present(
      y_average, "time_filter_interval", timeFilterInterval_,
      timeFilterInterval_);
    if (y_average["averaging_type"]) {
      std::string avgType = y_average["averaging_type"].as<std::string>();
      if (avgType == "nalu_classic")
        averagingType_ = NALU_CLASSIC;
      else if (avgType == "moving_exponential")
        averagingType_ = MOVING_EXPONENTIAL;
      else
        throw std::runtime_error(
          "TurbulenceAveragingPostProcessing: "
          "Invalid averaging type specified for turbulence post processing.");
    }

    // extract the sequence of types
    const YAML::Node y_specs =
      expect_sequence(y_average, "specifications", false);
    if (y_specs) {
      for (size_t ispec = 0; ispec < y_specs.size(); ++ispec) {
        const YAML::Node& y_spec = (y_specs)[ispec];

        // new the info object
        AveragingInfo* avInfo = new AveragingInfo();

        // find the name
        const YAML::Node theName = y_spec["name"];
        if (theName)
          avInfo->name_ = theName.as<std::string>();
        else
          throw std::runtime_error(
            "TurbulenceAveragingPostProcessing: no name provided");

        // extract the set of target names
        const YAML::Node targets = y_spec["target_name"];
        if (targets.Type() == YAML::NodeType::Scalar) {
          avInfo->targetNames_.resize(1);
          avInfo->targetNames_[0] = targets.as<std::string>();
        } else {
          avInfo->targetNames_.resize(targets.size());
          for (size_t i = 0; i < targets.size(); ++i) {
            avInfo->targetNames_[i] = targets[i].as<std::string>();
          }
        }

        // reynolds
        const YAML::Node y_reynolds = y_spec["reynolds_averaged_variables"];
        if (y_reynolds) {
          for (size_t ioption = 0; ioption < y_reynolds.size(); ++ioption) {
            const YAML::Node y_var = y_reynolds[ioption];
            std::string fieldName = y_var.as<std::string>();
            if (fieldName != "density")
              avInfo->reynoldsFieldNameVec_.push_back(fieldName);
          }
        }

        // Favre
        const YAML::Node y_favre = y_spec["favre_averaged_variables"];
        if (y_favre) {
          for (size_t ioption = 0; ioption < y_favre.size(); ++ioption) {
            const YAML::Node y_var = y_favre[ioption];
            std::string fieldName = y_var.as<std::string>();
            if (fieldName != "density")
              avInfo->favreFieldNameVec_.push_back(fieldName);
          }
        }

        const YAML::Node y_movavg = y_spec["moving_averaged_variables"];
        if (y_movavg) {
          for (size_t ioption = 0; ioption < y_movavg.size(); ++ioption) {
            const YAML::Node y_var = y_movavg[ioption];
            std::string fieldName = y_var.as<std::string>();
            avInfo->movingAvgFieldNameVec_.push_back(fieldName);
          }
        }

        // check for stress and tke post processing; Reynolds and Favre
        get_if_present(
          y_spec, "compute_reynolds_stress", avInfo->computeReynoldsStress_,
          avInfo->computeReynoldsStress_);
        get_if_present(
          y_spec, "compute_tke", avInfo->computeTke_, avInfo->computeTke_);
        get_if_present(
          y_spec, "compute_favre_stress", avInfo->computeFavreStress_,
          avInfo->computeFavreStress_);
        get_if_present(
          y_spec, "compute_resolved_stress", avInfo->computeResolvedStress_,
          avInfo->computeResolvedStress_);
        get_if_present(
          y_spec, "compute_sfs_stress", avInfo->computeSFSStress_,
          avInfo->computeSFSStress_);
        get_if_present(
          y_spec, "compute_favre_tke", avInfo->computeFavreTke_,
          avInfo->computeFavreTke_);
        get_if_present(
          y_spec, "compute_vorticity", avInfo->computeVorticity_,
          avInfo->computeVorticity_);
        get_if_present(
          y_spec, "compute_q_criterion", avInfo->computeQcriterion_,
          avInfo->computeQcriterion_);
        get_if_present(
          y_spec, "compute_lambda_ci", avInfo->computeLambdaCI_,
          avInfo->computeLambdaCI_);
        get_if_present(
          y_spec, "compute_mean_resolved_ke", avInfo->computeMeanResolvedKe_,
          avInfo->computeMeanResolvedKe_);

        get_if_present(
          y_spec, "compute_temperature_sfs_flux",
          avInfo->computeTemperatureSFS_, avInfo->computeTemperatureSFS_);
        get_if_present(
          y_spec, "compute_temperature_resolved_flux",
          avInfo->computeTemperatureResolved_,
          avInfo->computeTemperatureResolved_);

        // we will need Reynolds/Favre-averaged velocity if we need to compute
        // TKE
        if (avInfo->computeTke_ || avInfo->computeReynoldsStress_) {
          const std::string velocityName = "velocity";
          if (
            std::find(
              avInfo->reynoldsFieldNameVec_.begin(),
              avInfo->reynoldsFieldNameVec_.end(),
              velocityName) == avInfo->reynoldsFieldNameVec_.end()) {
            // not found; add it
            avInfo->reynoldsFieldNameVec_.push_back(velocityName);
          }
        }

        if (avInfo->computeFavreTke_ || avInfo->computeFavreStress_) {
          const std::string velocityName = "velocity";
          if (
            std::find(
              avInfo->favreFieldNameVec_.begin(),
              avInfo->favreFieldNameVec_.end(),
              velocityName) == avInfo->favreFieldNameVec_.end()) {
            // not found; add it
            avInfo->favreFieldNameVec_.push_back(velocityName);
          }
        }

        if (
          avInfo->computeResolvedStress_ ||
          avInfo->computeTemperatureResolved_) {
          const std::string velocityName = "velocity";
          if (
            std::find(
              avInfo->resolvedFieldNameVec_.begin(),
              avInfo->resolvedFieldNameVec_.end(),
              velocityName) == avInfo->resolvedFieldNameVec_.end()) {
            // not found; add it
            avInfo->resolvedFieldNameVec_.push_back(velocityName);
          }
        }

        if (avInfo->computeTemperatureResolved_) {
          const std::string temperatureName = "temperature";
          if (
            std::find(
              avInfo->resolvedFieldNameVec_.begin(),
              avInfo->resolvedFieldNameVec_.end(),
              temperatureName) == avInfo->resolvedFieldNameVec_.end()) {
            // not found; add it
            avInfo->resolvedFieldNameVec_.push_back(temperatureName);
          }
        }

        // push back the object
        averageInfoVec_.push_back(avInfo);
      }
    } else {
      throw std::runtime_error(
        "TurbulenceAveragingPostProcessing: no specifications provided");
    }
  }
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::setup()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  // Special case for boussinesq_ra algorithm
  // The algorithm requires that "temperature_ma" be available
  // on all blocks where temperature is defined.
  if (realm_.solutionOptions_->has_set_boussinesq_time_scale()) {
    const std::string temperatureName = "temperature";
    const std::string fTempName =
      MovingAveragePostProcessor::filtered_field_name(temperatureName);

    auto* tempField =
      metaData.get_field(stk::topology::NODE_RANK, "temperature");
    ThrowRequireMsg(
      tempField != nullptr, "Temperature field must be registered");

    auto& field = metaData.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, fTempName);
    stk::mesh::put_field_on_mesh(
      field, stk::mesh::selectField(*tempField), nullptr);
    realm_.augment_restart_variable_list(fTempName);

    movingAvgPP_ = std::make_unique<MovingAveragePostProcessor>(
      realm_.bulk_data(), *realm_.timeIntegrator_,
      realm_.restarted_simulation());
    movingAvgPP_->add_fields({temperatureName});
    movingAvgPP_->set_time_scale(
      realm_.solutionOptions_->raBoussinesqTimeScale_);
  }

  // loop over all info and setup (register fields, set parts, etc.)
  for (size_t k = 0; k < averageInfoVec_.size(); ++k) {

    // extract the turb info and the name
    AveragingInfo* avInfo = averageInfoVec_[k];

    const std::string averageBlockName = avInfo->name_;

    // loop over all target names, extract the part; register the fields
    for (size_t itarget = 0; itarget < avInfo->targetNames_.size(); ++itarget) {
      stk::mesh::Part* targetPart = metaData.get_part(
        realm_.physics_part_name(avInfo->targetNames_[itarget]));
      if (NULL == targetPart) {
        NaluEnv::self().naluOutputP0()
          << "Trouble with part " << avInfo->targetNames_[itarget] << std::endl;
        throw std::runtime_error(
          "Sorry, no part name found by the name: " +
          realm_.physics_part_name(avInfo->targetNames_[itarget]));
      } else {
        // push back
        avInfo->partVec_.push_back(targetPart);
      }

      // register special fields whose name prevails over the averaging info
      // name
      if (avInfo->computeTke_) {
        const std::string tkeName = "resolved_turbulent_ke";
        const int sizeOfField = 1;
        register_field(tkeName, sizeOfField, metaData, targetPart);
      }

      if (avInfo->computeFavreTke_) {
        const std::string tkeName = "resolved_favre_turbulent_ke";
        const int sizeOfField = 1;
        register_field(tkeName, sizeOfField, metaData, targetPart);
      }

      if (avInfo->computeVorticity_) {
        const int vortSize = realm_.spatialDimension_;
        const std::string vorticityName = "vorticity";
        VectorFieldType* vortField = &(metaData.declare_field<VectorFieldType>(
          stk::topology::NODE_RANK, vorticityName));
        stk::mesh::put_field_on_mesh(
          *vortField, *targetPart, vortSize, nullptr);
      }

      if (avInfo->computeQcriterion_) {
        const std::string QcritName = "q_criterion";
        const int sizeOfField = 1;
        register_field(QcritName, sizeOfField, metaData, targetPart);
      }

      if (avInfo->computeLambdaCI_) {
        const std::string lambdaName = "lambda_ci";
        const int sizeOfField = 1;
        register_field(lambdaName, sizeOfField, metaData, targetPart);
      }

      const int stressSize = realm_.spatialDimension_ == 3 ? 6 : 3;
      const int tempFluxSize = realm_.spatialDimension_;
      if (avInfo->computeReynoldsStress_) {
        const std::string stressName = "reynolds_stress";
        register_field(stressName, stressSize, metaData, targetPart);
      }

      if (avInfo->computeFavreStress_) {
        const std::string stressName = "favre_stress";
        register_field(stressName, stressSize, metaData, targetPart);
      }

      if (
        avInfo->computeResolvedStress_ || avInfo->computeTemperatureResolved_) {
        const std::string stressName = "resolved_stress";
        register_field(stressName, stressSize, metaData, targetPart);
      }

      if (avInfo->computeTemperatureResolved_) {
        const std::string tempFluxName = "temperature_resolved_flux";
        register_field(tempFluxName, tempFluxSize, metaData, targetPart);
        const std::string tempVarName = "temperature_variance";
        register_field(tempVarName, 1, metaData, targetPart);
      }

      if (avInfo->computeSFSStress_ || avInfo->computeTemperatureSFS_) {
        if (realm_.spatialDimension_ < 3)
          throw std::runtime_error(
            "TurbulenceAveragingPostProcessing:setup() Cannot compute SFS "
            "stress in less than 3 dimensions: ");
        const std::string stressName = "sfs_stress";
        register_field(stressName, stressSize, metaData, targetPart);
        const std::string SFSstressNameInst = "sfs_stress_inst";
        register_field(SFSstressNameInst, stressSize, metaData, targetPart);
      }

      if (avInfo->computeTemperatureSFS_) {
        const std::string tempFluxName = "temperature_sfs_flux";
        register_field(tempFluxName, tempFluxSize, metaData, targetPart);
      }

      // deal with density; always need Reynolds averaged quantity
      const std::string densityReynoldsName = "density_ra_" + averageBlockName;
      ScalarFieldType* densityReynolds =
        &(metaData.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, densityReynoldsName));
      stk::mesh::put_field_on_mesh(*densityReynolds, *targetPart, nullptr);

      // Reynolds
      for (size_t i = 0; i < avInfo->reynoldsFieldNameVec_.size(); ++i) {
        const std::string primitiveName = avInfo->reynoldsFieldNameVec_[i];
        const std::string averagedName =
          primitiveName + "_ra_" + averageBlockName;
        register_field_from_primitive(
          primitiveName, averagedName, metaData, targetPart);
      }

      // Favre
      for (size_t i = 0; i < avInfo->favreFieldNameVec_.size(); ++i) {
        const std::string primitiveName = avInfo->favreFieldNameVec_[i];
        const std::string averagedName =
          primitiveName + "_fa_" + averageBlockName;
        register_field_from_primitive(
          primitiveName, averagedName, metaData, targetPart);
      }

      // Resolved
      for (size_t i = 0; i < avInfo->resolvedFieldNameVec_.size(); ++i) {
        const std::string primitiveName = avInfo->resolvedFieldNameVec_[i];
        const std::string averagedName =
          primitiveName + "_resa_" + averageBlockName;
        register_field_from_primitive(
          primitiveName, averagedName, metaData, targetPart);
      }
    }

    // now deal with pairs; extract density
    const std::string densityName = "density";
    const std::string densityReynoldsName = "density_ra_" + averageBlockName;
    stk::mesh::FieldBase* density =
      metaData.get_field(stk::topology::NODE_RANK, densityName);
    stk::mesh::FieldBase* densityReynolds =
      metaData.get_field(stk::topology::NODE_RANK, densityReynoldsName);
    avInfo->reynoldsFieldVecPair_.push_back(
      std::make_pair(density, densityReynolds));
    avInfo->reynoldsFieldSizeVec_.push_back(1);
    realm_.augment_restart_variable_list(densityReynoldsName);

    // Reynolds
    for (size_t i = 0; i < avInfo->reynoldsFieldNameVec_.size(); ++i) {
      const std::string primitiveName = avInfo->reynoldsFieldNameVec_[i];
      const std::string averagedName =
        primitiveName + "_ra_" + averageBlockName;
      construct_pair(
        primitiveName, averagedName, avInfo->reynoldsFieldVecPair_,
        avInfo->reynoldsFieldSizeVec_, metaData);
    }

    // Favre
    for (size_t i = 0; i < avInfo->favreFieldNameVec_.size(); ++i) {
      const std::string primitiveName = avInfo->favreFieldNameVec_[i];
      const std::string averagedName =
        primitiveName + "_fa_" + averageBlockName;
      construct_pair(
        primitiveName, averagedName, avInfo->favreFieldVecPair_,
        avInfo->favreFieldSizeVec_, metaData);
    }

    // Resolved
    for (size_t i = 0; i < avInfo->resolvedFieldNameVec_.size(); ++i) {
      const std::string primitiveName = avInfo->resolvedFieldNameVec_[i];
      const std::string averagedName =
        primitiveName + "_resa_" + averageBlockName;
      construct_pair(
        primitiveName, averagedName, avInfo->resolvedFieldVecPair_,
        avInfo->resolvedFieldSizeVec_, metaData);
    }

    // output what we have done here...
    review(avInfo);
  }
}

//--------------------------------------------------------------------------
//-------- register_field_from_primitive -----------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::register_field_from_primitive(
  const std::string primitiveName,
  const std::string averagedName,
  stk::mesh::MetaData& metaData,
  stk::mesh::Part* part)
{
  // first, augment the restart list
  realm_.augment_restart_variable_list(averagedName);

  // declare field; put the field and augment restart; need size from the
  // primitive
  stk::mesh::FieldBase* primitiveField =
    metaData.get_field(stk::topology::NODE_RANK, primitiveName);

  // check for existence and if it is a double
  if (NULL == primitiveField)
    throw std::runtime_error(
      "TurbulenceAveragingPostProcessing::register_field() no primitive by "
      "this name: " +
      primitiveName);

  if (!primitiveField->type_is<double>())
    throw std::runtime_error(
      "TurbulenceAveragingPostProcessing::register_field() type of field is "
      "not double: " +
      primitiveName);

  // extract size (would love to do this by part), however, not yet a use case
  const unsigned fieldSizePrimitive =
    primitiveField->max_size(stk::topology::NODE_RANK);

  // register the averaged field with this size; treat velocity as a special
  // case to retain the vector aspect
  if (primitiveName == "velocity") {
    VectorFieldType* averagedField = &(metaData.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, averagedName));
    stk::mesh::put_field_on_mesh(
      *averagedField, *part, fieldSizePrimitive, nullptr);
  } else {
    stk::mesh::FieldBase* averagedField =
      &(metaData
          .declare_field<stk::mesh::Field<double, stk::mesh::SimpleArrayTag>>(
            stk::topology::NODE_RANK, averagedName));
    stk::mesh::put_field_on_mesh(
      *averagedField, *part, fieldSizePrimitive, nullptr);
  }
}

//--------------------------------------------------------------------------
//-------- construct_pair --------------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::construct_pair(
  const std::string primitiveName,
  const std::string averagedName,
  std::vector<std::pair<stk::mesh::FieldBase*, stk::mesh::FieldBase*>>&
    fieldVecPair,
  std::vector<unsigned>& fieldSizeVec,
  stk::mesh::MetaData& metaData)
{
  // augment the restart list
  realm_.augment_restart_variable_list(averagedName);

  // extract the valid primitive and averaged field
  stk::mesh::FieldBase* primitiveField =
    metaData.get_field(stk::topology::NODE_RANK, primitiveName);
  stk::mesh::FieldBase* averagedField =
    metaData.get_field(stk::topology::NODE_RANK, averagedName);

  // the size; guaranteed to be the same based on the field registration
  const unsigned fieldSizeAveraged =
    averagedField->max_size(stk::topology::NODE_RANK);
  fieldSizeVec.push_back(fieldSizeAveraged);

  // construct pairs
  fieldVecPair.push_back(std::make_pair(primitiveField, averagedField));
}

//--------------------------------------------------------------------------
//-------- register_field --------------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::register_field(
  const std::string fieldName,
  const int fieldSize,
  stk::mesh::MetaData& metaData,
  stk::mesh::Part* targetPart)
{
  // register and put the field
  stk::mesh::FieldBase* theField = &(
    metaData.declare_field<stk::mesh::Field<double, stk::mesh::SimpleArrayTag>>(
      stk::topology::NODE_RANK, fieldName));
  stk::mesh::put_field_on_mesh(*theField, *targetPart, fieldSize, nullptr);
  // augment the restart list
  realm_.augment_restart_variable_list(fieldName);
}

//--------------------------------------------------------------------------
//-------- review ----------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::review(const AveragingInfo* avInfo)
{
  // review what will be done
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Averaging Review: " << avInfo->name_ << std::endl;
  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;
  for (size_t iav = 0; iav < avInfo->reynoldsFieldVecPair_.size(); ++iav) {
    stk::mesh::FieldBase* primitiveFB =
      avInfo->reynoldsFieldVecPair_[iav].first;
    stk::mesh::FieldBase* averageFB = avInfo->reynoldsFieldVecPair_[iav].second;
    NaluEnv::self().naluOutputP0()
      << "Primitive/Reynolds name: " << primitiveFB->name() << "/"
      << averageFB->name() << " size " << avInfo->reynoldsFieldSizeVec_[iav]
      << std::endl;
  }

  for (size_t iav = 0; iav < avInfo->favreFieldVecPair_.size(); ++iav) {
    stk::mesh::FieldBase* primitiveFB = avInfo->favreFieldVecPair_[iav].first;
    stk::mesh::FieldBase* averageFB = avInfo->favreFieldVecPair_[iav].second;
    NaluEnv::self().naluOutputP0()
      << "Primitive/Favre name:    " << primitiveFB->name() << "/"
      << averageFB->name() << " size " << avInfo->favreFieldSizeVec_[iav]
      << std::endl;
  }

  for (size_t iav = 0; iav < avInfo->resolvedFieldVecPair_.size(); ++iav) {
    stk::mesh::FieldBase* primitiveFB =
      avInfo->resolvedFieldVecPair_[iav].first;
    stk::mesh::FieldBase* averageFB = avInfo->resolvedFieldVecPair_[iav].second;
    NaluEnv::self().naluOutputP0()
      << "Primitive/Resolved name: " << primitiveFB->name() << "/"
      << averageFB->name() << " size " << avInfo->resolvedFieldSizeVec_[iav]
      << std::endl;
  }

  if (movingAvgPP_ != nullptr) {
    for (const auto& fieldPair : movingAvgPP_->get_field_map()) {
      stk::mesh::FieldBase* primitiveFB = fieldPair.first;
      stk::mesh::FieldBase* averageFB = fieldPair.second;
      NaluEnv::self().naluOutputP0()
        << "Primitive/Favre name:    " << primitiveFB->name() << "/"
        << averageFB->name() << std::endl;
    }
  }

  if (avInfo->computeTke_) {
    NaluEnv::self().naluOutputP0()
      << "TKE will be computed; add resolved_turbulent_ke to the "
         "Reynolds/Favre block for mean"
      << std::endl;
  }

  if (avInfo->computeFavreTke_) {
    NaluEnv::self().naluOutputP0()
      << "Favre-TKE will be computed; add resolved_favre_turbulent_ke to the "
         "Reynolds/Favre block for mean"
      << std::endl;
  }

  if (avInfo->computeReynoldsStress_) {
    NaluEnv::self().naluOutputP0()
      << "Reynolds Stress will be computed; add reynolds_stress to output"
      << std::endl;
  }

  if (avInfo->computeFavreStress_) {
    NaluEnv::self().naluOutputP0()
      << "Favre Stress will be computed; add favre_stress to output"
      << std::endl;
  }

  if (avInfo->computeResolvedStress_) {
    NaluEnv::self().naluOutputP0()
      << "Resolved Stress will be computed; add resolved_stress to output"
      << std::endl;
  }

  if (avInfo->computeSFSStress_) {
    NaluEnv::self().naluOutputP0()
      << "Sub-filter scale Stress will be computed; add sfs_stress to output"
      << std::endl;
  }

  if (avInfo->computeVorticity_) {
    NaluEnv::self().naluOutputP0()
      << "Vorticity will be computed; add vorticity to output" << std::endl;
  }

  if (avInfo->computeQcriterion_) {
    NaluEnv::self().naluOutputP0()
      << "Q criterion will be computed; add q_criterion to output" << std::endl;
  }

  if (avInfo->computeLambdaCI_) {
    NaluEnv::self().naluOutputP0()
      << "Lambda CI will be computed; add lambda_ci to output" << std::endl;
  }

  if (avInfo->computeMeanResolvedKe_) {
    NaluEnv::self().naluOutputP0()
      << "Mean resolved kinetic energy will be computed" << std::endl;
  }

  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::execute()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const double dt = realm_.get_time_step();
  double oldTimeFilter = currentTimeFilter_;
  double zeroCurrent = 1.0;

  if (averagingType_ == NALU_CLASSIC) {
    const bool resetFilter =
      (oldTimeFilter + dt > timeFilterInterval_) || forcedReset_;
    zeroCurrent = resetFilter ? 0.0 : 1.0;
    currentTimeFilter_ = resetFilter ? dt : oldTimeFilter + dt;
    NaluEnv::self().naluOutputP0()
      << "Filter Size " << currentTimeFilter_ << std::endl;
  } else if (averagingType_ == MOVING_EXPONENTIAL) {
    const double timeFilter = oldTimeFilter + dt;

    if (timeFilter > timeFilterInterval_) {
      currentTimeFilter_ = timeFilterInterval_;
      oldTimeFilter = timeFilterInterval_ - dt;
    } else {
      currentTimeFilter_ = timeFilter;
    }
    zeroCurrent = forcedReset_ ? 0.0 : 1.0;
  }

  // deactivate hard reset
  forcedReset_ = false;

  if (movingAvgPP_ != nullptr) {
    movingAvgPP_->execute();
  }

  // loop over all info and setup (register fields, set parts, etc.)
  for (size_t k = 0; k < averageInfoVec_.size(); ++k) {

    // extract the turb info and the name
    AveragingInfo* avInfo = averageInfoVec_[k];

    // define some common selectors
    stk::mesh::Selector s_all_nodes =
      (metaData.locally_owned_part() | metaData.globally_shared_part()) &
      stk::mesh::selectUnion(avInfo->partVec_) &
      !(realm_.get_inactive_selector());

    compute_averages(avInfo, s_all_nodes, oldTimeFilter, zeroCurrent, dt);

    // process special fields; internal avInfo flag defines the field
    if (avInfo->computeTke_) {
      compute_tke(true, avInfo->name_, s_all_nodes);
    }

    if (avInfo->computeFavreTke_) {
      compute_tke(false, avInfo->name_, s_all_nodes);
    }

    if (avInfo->computeVorticity_) {
      compute_vorticity(avInfo->name_, s_all_nodes);
    }

    if (avInfo->computeQcriterion_) {
      compute_q_criterion(avInfo->name_, s_all_nodes);
    }

    if (avInfo->computeLambdaCI_) {
      compute_lambda_ci(avInfo->name_, s_all_nodes);
    }

    if (avInfo->computeMeanResolvedKe_) {
      // need locally owned and active nodes
      stk::mesh::Selector s_locally_owned_nodes =
        metaData.locally_owned_part() &
        stk::mesh::selectUnion(avInfo->partVec_) &
        !(realm_.get_inactive_selector()) &
        !(stk::mesh::selectUnion(realm_.get_slave_part_vector()));
      compute_mean_resolved_ke(avInfo->name_, s_locally_owned_nodes);
    }

    // avoid computing stresses when when oldTimeFilter is not zero
    // this will occur only on a first time step of a new simulation
    if (oldTimeFilter > 0.0) {
      if (avInfo->computeFavreStress_) {
        compute_favre_stress(
          avInfo->name_, oldTimeFilter, zeroCurrent, dt, s_all_nodes);
      }

      if (avInfo->computeReynoldsStress_) {
        compute_reynolds_stress(
          avInfo->name_, oldTimeFilter, zeroCurrent, dt, s_all_nodes);
      }
    }
    if (avInfo->computeResolvedStress_) {
      compute_resolved_stress(
        avInfo->name_, oldTimeFilter, zeroCurrent, dt, s_all_nodes);
    }

    if (avInfo->computeSFSStress_) {
      compute_sfs_stress(
        avInfo->name_, oldTimeFilter, zeroCurrent, dt, s_all_nodes);
    }

    if (avInfo->computeTemperatureResolved_)
      compute_temperature_resolved_flux(
        avInfo->name_, oldTimeFilter, zeroCurrent, dt, s_all_nodes);

    if (avInfo->computeTemperatureSFS_)
      compute_temperature_sfs_flux(
        avInfo->name_, oldTimeFilter, zeroCurrent, dt, s_all_nodes);
  }
}

void
TurbulenceAveragingPostProcessing::compute_averages(
  AveragingInfo* avInfo,
  stk::mesh::Selector sel,
  const double& oldTimeFilter,
  const double& zeroCurrent,
  const double& dt)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  using FieldPair = Kokkos::pair<FieldInfoNGP, FieldInfoNGP>;
  using FieldInfoView = Kokkos::View<FieldPair*, Kokkos::LayoutRight, MemSpace>;

  const int numRePairs = avInfo->reynoldsFieldVecPair_.size();
  const int numFavrePairs = avInfo->favreFieldVecPair_.size();
  const int numResolvedPairs = avInfo->resolvedFieldVecPair_.size();
  const double currentTimeFilter = currentTimeFilter_;

#if defined(KOKKOS_ENABLE_GPU)
  FieldInfoView fieldPairs(
    Kokkos::ViewAllocateWithoutInitializing("turbAveragesFields"),
    (numRePairs + numFavrePairs + numResolvedPairs));
#else
  FieldInfoView fieldPairs(
    "turbAveragesFields", (numRePairs + numFavrePairs + numResolvedPairs));
#endif
  auto hostFieldPairs = Kokkos::create_mirror_view(fieldPairs);

  for (int i = 0; i < numRePairs; i++) {
    hostFieldPairs[i] = FieldPair(
      FieldInfoNGP(
        avInfo->reynoldsFieldVecPair_[i].first,
        avInfo->reynoldsFieldSizeVec_[i]),
      FieldInfoNGP(
        avInfo->reynoldsFieldVecPair_[i].second,
        avInfo->reynoldsFieldSizeVec_[i]));
  }

  int offset = numRePairs;
  for (int i = 0; i < numFavrePairs; i++) {
    hostFieldPairs[offset + i] = FieldPair(
      FieldInfoNGP(
        avInfo->favreFieldVecPair_[i].first, avInfo->favreFieldSizeVec_[i]),
      FieldInfoNGP(
        avInfo->favreFieldVecPair_[i].second, avInfo->favreFieldSizeVec_[i]));
  }

  offset += numFavrePairs;
  for (int i = 0; i < numResolvedPairs; i++) {
    hostFieldPairs[offset + i] = FieldPair(
      FieldInfoNGP(
        avInfo->resolvedFieldVecPair_[i].first,
        avInfo->resolvedFieldSizeVec_[i]),
      FieldInfoNGP(
        avInfo->resolvedFieldVecPair_[i].second,
        avInfo->resolvedFieldSizeVec_[i]));
  }
  Kokkos::deep_copy(fieldPairs, hostFieldPairs);

  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto density = fieldMgr.get_field<double>(
    avInfo->reynoldsFieldVecPair_[0].first->mesh_meta_data_ordinal());
  const auto densityA = fieldMgr.get_field<double>(
    avInfo->reynoldsFieldVecPair_[0].second->mesh_meta_data_ordinal());

  nalu_ngp::run_entity_algorithm(
    "TurbPP::compute_averages", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double oldRhoRA = densityA.get(mi, 0);
      const double rho = density.get(mi, 0);

      // Process reynolds averaging quantities first; used in Favre
      for (int i = 0; i < numRePairs; ++i) {
        const auto prim = fieldPairs(i).first.field;
        auto avg = fieldPairs(i).second.field;
        const auto numComponents = fieldPairs(i).first.scalarsDim1;

        for (unsigned j = 0; j < numComponents; ++j) {
          const double avgVal = (avg.get(mi, j) * oldTimeFilter * zeroCurrent +
                                 prim.get(mi, j) * dt) /
                                currentTimeFilter;
          avg.get(mi, j) = avgVal;
        }
      }

      // Favre averaged quantities
      int offset = numRePairs;
      const double rhoRA = densityA.get(mi, 0);
      for (int i = 0; i < numFavrePairs; ++i) {
        const int idx = offset + i;
        const auto prim = fieldPairs(idx).first.field;
        auto avg = fieldPairs(idx).second.field;
        const auto numComponents = fieldPairs(idx).first.scalarsDim1;

        for (unsigned j = 0; j < numComponents; ++j) {
          const double avgVal =
            (avg.get(mi, j) * oldRhoRA * oldTimeFilter * zeroCurrent +
             prim.get(mi, j) * rho * dt) /
            (currentTimeFilter * rhoRA);
          avg.get(mi, j) = avgVal;
        }
      }

      // Resolved quantities
      offset += numFavrePairs;
      for (int i = 0; i < numResolvedPairs; ++i) {
        const int idx = offset + i;
        const auto prim = fieldPairs(idx).first.field;
        auto avg = fieldPairs(idx).second.field;
        const auto numComponents = fieldPairs(idx).first.scalarsDim1;

        for (unsigned j = 0; j < numComponents; ++j) {
          const double avgVal = (avg.get(mi, j) * oldTimeFilter * zeroCurrent +
                                 rho * prim.get(mi, j) * dt) /
                                currentTimeFilter;
          avg.get(mi, j) = avgVal;
        }
      }
    });

  {
    // Tag fields as modified on device
    const auto numNGPFields = hostFieldPairs.extent(0);
    for (unsigned i = 0; i < numNGPFields; ++i) {
      auto& field = hostFieldPairs(i).second.field;
      field.modify_on_device();
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_tke -----------------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_tke(
  const bool isReynolds,
  const std::string& averageBlockName,
  stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  // check for precise set of names
  const std::string velocityName = isReynolds
                                     ? "velocity_ra_" + averageBlockName
                                     : "velocity_fa_" + averageBlockName;
  const std::string resolvedTkeName =
    isReynolds ? "resolved_turbulent_ke" : "resolved_favre_turbulent_ke";

  const int ndim = realm_.meta_data().spatial_dimension();
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  const auto velocityA = nalu_ngp::get_ngp_field(meshInfo, velocityName);
  auto resTKE = nalu_ngp::get_ngp_field(meshInfo, resolvedTkeName);

  nalu_ngp::run_entity_algorithm(
    "TurbPP::compute_tke", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      double sum = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const double uprime = velocity.get(mi, d) - velocityA.get(mi, d);
        sum += 0.5 * uprime * uprime;
      }
      resTKE.get(mi, 0) = sum;
    });
  resTKE.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- compute_reynolds_stress -----------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_reynolds_stress(
  const std::string& averageBlockName,
  const double& oldTimeFilter,
  const double& zeroCurrent,
  const double& dt,
  stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const std::string velocityAName = "velocity_ra_" + averageBlockName;
  const std::string stressName = "reynolds_stress";

  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  const auto velocityA = nalu_ngp::get_ngp_field(meshInfo, velocityAName);
  auto stress = nalu_ngp::get_ngp_field(meshInfo, stressName);

  const double oldWeight = oldTimeFilter * zeroCurrent;
  const double currentTimeFilter = currentTimeFilter_;

  stress.sync_to_device();
  nalu_ngp::run_entity_algorithm(
    "TurbPP::compute_restress", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      int ic = 0;

      for (int i = 0; i < ndim; ++i) {
        const double ui = velocity.get(mi, i);
        const double uAi = velocityA.get(mi, i);
        const double uAiOld =
          (currentTimeFilter * uAi - ui * dt) / oldTimeFilter;

        for (int j = i; j < ndim; ++j) {
          const double uj = velocity.get(mi, j);
          const double uAj = velocityA.get(mi, j);
          const double uAjOld =
            (currentTimeFilter * uAj - uj * dt) / oldTimeFilter;

          const double stressVal =
            ((stress.get(mi, ic) + uAiOld * uAjOld) * oldWeight +
             ui * uj * dt) /
              currentTimeFilter -
            uAi * uAj;

          stress.get(mi, ic) = stressVal;
          ic++;
        }
      }
    });
  stress.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- compute_favre_stress --------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_favre_stress(
  const std::string& averageBlockName,
  const double& oldTimeFilter,
  const double& zeroCurrent,
  const double& dt,
  stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const std::string velocityAName = "velocity_fa_" + averageBlockName;
  const std::string densityAName = "density_ra_" + averageBlockName;
  const std::string stressName = "favre_stress";

  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto densityA = nalu_ngp::get_ngp_field(meshInfo, densityAName);
  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  const auto velocityA = nalu_ngp::get_ngp_field(meshInfo, velocityAName);
  auto stress = nalu_ngp::get_ngp_field(meshInfo, stressName);

  const double currentTimeFilter = currentTimeFilter_;

  nalu_ngp::run_entity_algorithm(
    "TurbPP::compute_favre_stress", ngpMesh, stk::topology::NODE_RANK,
    s_all_nodes, KOKKOS_LAMBDA(const MeshIndex& mi) {
      int ic = 0;

      const double rho = density.get(mi, 0);
      const double rhoA = densityA.get(mi, 0);
      const double rhoAOld =
        (currentTimeFilter * rhoA - rho * dt) / oldTimeFilter;

      const double rAOldbyRA = rhoAOld / rhoA;
      const double rbyRA = rho / rhoA;

      for (int i = 0; i < ndim; ++i) {
        const double ui = velocity.get(mi, i);
        const double uAi = velocityA.get(mi, i);
        const double uAiOld = (currentTimeFilter * rhoA * uAi - rho * ui * dt) /
                              (oldTimeFilter * rhoAOld);

        for (int j = i; j < ndim; ++j) {
          const double uj = velocity.get(mi, j);
          const double uAj = velocityA.get(mi, j);
          const double uAjOld =
            (currentTimeFilter * rhoA * uAj - rho * uj * dt) /
            (oldTimeFilter * rhoAOld);

          const double stressVal = ((stress.get(mi, ic) + uAiOld * uAjOld) *
                                      rAOldbyRA * oldTimeFilter * zeroCurrent +
                                    rbyRA * ui * uj * dt) /
                                     currentTimeFilter -
                                   uAi * uAj;

          stress.get(mi, ic) = stressVal;
          ic++;
        }
      }
    });
  stress.modify_on_device();
}

void
TurbulenceAveragingPostProcessing::compute_temperature_resolved_flux(
  const std::string&,
  const double& oldTimeFilter,
  const double& zeroCurrent,
  const double& dt,
  stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.meta_data().spatial_dimension();
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  const auto density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto temperature = nalu_ngp::get_ngp_field(meshInfo, "temperature");
  auto tempFlux =
    nalu_ngp::get_ngp_field(meshInfo, "temperature_resolved_flux");
  auto tempVar = nalu_ngp::get_ngp_field(meshInfo, "temperature_variance");

  const double currentTimeFilter = currentTimeFilter_;

  nalu_ngp::run_entity_algorithm(
    "TurbPP::temp_res_flux", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double rho = density.get(mi, 0);
      const double temp = temperature.get(mi, 0);
      const double tvar = tempVar.get(mi, 0);

      tempVar.get(mi, 0) =
        (tvar * oldTimeFilter * zeroCurrent + rho * temp * temp * dt) /
        currentTimeFilter;

      for (int d = 0; d < ndim; ++d) {
        const double ui = velocity.get(mi, d);
        const double tflux = tempFlux.get(mi, d);

        tempFlux.get(mi, d) =
          (tflux * oldTimeFilter * zeroCurrent + rho * ui * temp * dt) /
          currentTimeFilter;
      }
    });

  tempFlux.modify_on_device();
  tempVar.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- compute_resolved_stress -----------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_resolved_stress(
  const std::string&,
  const double& oldTimeFilter,
  const double& zeroCurrent,
  const double& dt,
  stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  auto stress = nalu_ngp::get_ngp_field(meshInfo, "resolved_stress");

  const double currentTimeFilter = currentTimeFilter_;

  nalu_ngp::run_entity_algorithm(
    "TurbPP::resolved_stress", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      int ic = 0;

      const double rho = density.get(mi, 0);
      for (int i = 0; i < ndim; ++i) {
        const double ui = velocity.get(mi, i);

        for (int j = i; j < ndim; ++j) {
          const double uj = velocity.get(mi, j);
          const double newStress =
            (stress.get(mi, ic) * oldTimeFilter * zeroCurrent +
             rho * ui * uj * dt) /
            currentTimeFilter;

          stress.get(mi, ic) = newStress;
          ic++;
        }
      }
    });
  stress.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- compute_vortictiy -----------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_sfs_stress(
  const std::string& /* averageBlockName */,
  const double& oldTimeFilter,
  const double& zeroCurrent,
  const double& dt,
  stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const double twoDivDim = 2.0 / static_cast<double>(ndim);
  const double twothird = 2.0 / 3.0;

  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto dualVol = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");
  const auto turbVisc =
    nalu_ngp::get_ngp_field(meshInfo, "turbulent_viscosity");
  const auto dudx = nalu_ngp::get_ngp_field(meshInfo, "dudx");
  auto sfsStress = nalu_ngp::get_ngp_field(meshInfo, "sfs_stress");
  auto sfsStressInst = nalu_ngp::get_ngp_field(meshInfo, "sfs_stress_inst");

  // Special treatment for turbulent KE
  const auto* turbKEHost =
    realm_.meta_data().get_field(stk::topology::NODE_RANK, "turbulent_ke");
  const bool computeSFSTKE = (turbKEHost == nullptr);

  // If we have a turbulent_ke field, extract the NGP version for use in
  // computations
  stk::mesh::NgpField<double> turbKE;
  if (!computeSFSTKE) {
    turbKE = nalu_ngp::get_ngp_field(meshInfo, "turbulent_ke");
  }

  const double tm_ci = realm_.get_turb_model_constant(TM_ci);
  const double currentTimeFilter = currentTimeFilter_;

  nalu_ngp::run_entity_algorithm(
    "TurbPP::sfs_stress", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      double divU = 0.0;
      for (int d = 0; d < ndim; ++d)
        divU += dudx.get(mi, ndim * d + d);

      double sfsTKE = 0.0;
      const double rho = density.get(mi, 0);
      const double mut = turbVisc.get(mi, 0);

      if (computeSFSTKE) {
        //
        // Turbulent KE field not available. Compute SFS TKE term using method
        // of Yoshizawa (1986), "Statistical theory for compressible turbulent
        // shear flows, with the application to subgrid modeling",
        // https://doi.org/10.1063/1.865552
        //
        double sijmagsq = 0.0;
        for (int i = 0; i < ndim; ++i)
          for (int j = 0; j < ndim; ++j) {
            const double rateOfStrain =
              0.5 * (dudx.get(mi, ndim * i + j) + dudx.get(mi, ndim * j + i));
            sijmagsq += rateOfStrain * rateOfStrain;
          }
        sfsTKE = tm_ci * stk::math::pow(dualVol.get(mi, 0), twoDivDim) *
                 (2.0 * sijmagsq);
      } else {
        sfsTKE = turbKE.get(mi, 0);
      }

      int ic = 0;
      for (int i = 0; i < ndim; ++i)
        for (int j = i; j < ndim; ++j) {
          const double divUTerm = (i == j) ? twothird * divU : 0.0;
          const double sfsTKETerm = (i == j) ? twothird * rho * sfsTKE : 0.0;

          const double instStress =
            -(mut * (dudx.get(mi, ndim * i + j) + dudx.get(mi, ndim * j + i) -
                     divUTerm) -
              sfsTKETerm);
          sfsStressInst.get(mi, ic) = instStress;
          const double newStress =
            (sfsStress.get(mi, ic) * oldTimeFilter * zeroCurrent -
             dt * (mut * (dudx.get(mi, ndim * i + j) +
                          dudx.get(mi, ndim * j + i) - divUTerm) -
                   sfsTKETerm)) /
            currentTimeFilter;
          sfsStress.get(mi, ic) = newStress;
          ic++;
        }
    });
  sfsStress.modify_on_device();
  sfsStressInst.modify_on_device();
}

void
TurbulenceAveragingPostProcessing::compute_temperature_sfs_flux(
  const std::string&,
  const double& oldTimeFilter,
  const double& zeroCurrent,
  const double& dt,
  stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const double turbPr = realm_.get_turb_prandtl("enthalpy");
  const double currentTimeFilter = currentTimeFilter_;

  const auto turbVisc =
    nalu_ngp::get_ngp_field(meshInfo, "turbulent_viscosity");
  const auto dhdx = nalu_ngp::get_ngp_field(meshInfo, "dhdx");
  const auto specHeat = nalu_ngp::get_ngp_field(meshInfo, "specific_heat");
  auto tempSfsFlux = nalu_ngp::get_ngp_field(meshInfo, "temperature_sfs_flux");

  nalu_ngp::run_entity_algorithm(
    "TurbPP::temp_sfs_flux", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const double nut = turbVisc.get(mi, 0);
      const double cp = specHeat.get(mi, 0);

      for (int d = 0; d < ndim; ++d) {
        const double tempFlux =
          (tempSfsFlux.get(mi, d) * oldTimeFilter * zeroCurrent -
           dt * nut / (turbPr * cp) * dhdx.get(mi, d)) /
          currentTimeFilter;
        tempSfsFlux.get(mi, d) = tempFlux;
      }
    });
  tempSfsFlux.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- compute_vortictiy -----------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_vorticity(
  const std::string& /* averageBlockName */, stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();

  const auto dudx = nalu_ngp::get_ngp_field(meshInfo, "dudx");
  auto vort = nalu_ngp::get_ngp_field(meshInfo, "vorticity");

  nalu_ngp::run_entity_algorithm(
    "TurbPP::vorticity", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      for (int i = 0; i < ndim; ++i) {
        // (i, j) = (0, 1) or (1, 2) or (2, 0)
        const int j = (i + 1) % ndim;

        vort.get(mi, ndim - i - j) =
          dudx.get(mi, ndim * j + i) - dudx.get(mi, ndim * i + j);
      }
    });
  vort.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- compute_q_criterion----------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_q_criterion(
  const std::string& /* averageBlockName */, stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();

  const auto dudx = nalu_ngp::get_ngp_field(meshInfo, "dudx");
  auto qcrit = nalu_ngp::get_ngp_field(meshInfo, "q_criterion");

  nalu_ngp::run_entity_algorithm(
    "TurbPP::q_crit", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      double sij = 0.0;
      double vort = 0.0;

      for (int i = 0; i < ndim; ++i)
        for (int j = 0; j < ndim; ++j) {
          const double duidxj = dudx.get(mi, ndim * i + j);
          const double dujdxi = dudx.get(mi, ndim * j + i);

          const double rateOfStrain = 0.5 * (duidxj + dujdxi);
          const double vortTensor = 0.5 * (duidxj - dujdxi);
          sij += rateOfStrain * rateOfStrain;
          vort += vortTensor * vortTensor;
        }

      double divSqr = 0.0;
      if (ndim == 2) {
        const double div = dudx.get(mi, 0) + dudx.get(mi, 3);
        divSqr = div * div;
      } else {
        const double div = dudx.get(mi, 0) + dudx.get(mi, 4) + dudx.get(mi, 8);
        divSqr = div * div;
      }

      qcrit.get(mi, 0) = 0.5 * (vort - sij + divSqr);
    });

  qcrit.modify_on_device();
}

//--------------------------------------------------------------------------
//-------- compute_lambda_ci -----------------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_lambda_ci(
  const std::string& /* averageBlockName */, stk::mesh::Selector s_all_nodes)
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  const int nDim = realm_.spatialDimension_;
  const std::string lambdaName = "lambda_ci";

  // extract fields
  stk::mesh::FieldBase* Lambda =
    metaData.get_field(stk::topology::NODE_RANK, lambdaName);
  GenericFieldType* dudx_ =
    metaData.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");

  stk::mesh::BucketVector const& node_buckets_vort =
    realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets_vort.begin();
       ib != node_buckets_vort.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // fields
    double* LambdaCI_ = (double*)stk::mesh::field_data(*Lambda, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      stk::mesh::Entity node = b[k];

      const double* a_matrix = stk::mesh::field_data(*dudx_, node);

      // Check if 2-D or 3-D, which will determine whether to solve a quadratic
      // or cubic equation
      if (nDim == 2) {
        // Solve a quadratic eigenvalue equation, A*Lambda^2 + B*Lambda + C = 0
        const double a11 = a_matrix[0];
        const double a12 = a_matrix[1];
        const double a21 = a_matrix[2];
        const double a22 = a_matrix[3];

        // For a 2x2 matrix, the first and second invariant are the -trace and
        // the determinant
        const double trace = a11 + a22;
        const double det = a11 * a22 - a12 * a21;

        std::complex<double> A(1.0, 0.0);
        const double Ar = 1.0;
        std::complex<double> B(-trace, 0.0);
        const double Br = -trace;
        std::complex<double> C(det, 0.0);
        const double Cr = det;
        const double Discrim = Br * Br - 4 * Ar * Cr;

        // Check whether real or complex eigenvalues
        if (Discrim >= 0) {
          // Two real eigenvalues, lambda_ci not applicable
          LambdaCI_[k] = 0.0;
        } else {
          // Two complex conjugate eigenvalues, lambda_ci applicable
          std::complex<double> EIG1;
          EIG1 = -B / 2.0 + std::sqrt(B * B - A * C * 4.0) / 2.0;
          std::complex<double> EIG2;
          EIG2 = -B / 2.0 - std::sqrt(B * B - A * C * 4.0) / 2.0;
          LambdaCI_[k] = std::max(std::imag(EIG1), std::imag(EIG2));
        }
      } else {
        // Solve a cubic eigenvalue equation, A*Lambda^3 + B*Lambda^2 + C*Lambda
        // + D = 0
        const double a11 = a_matrix[0];
        const double a12 = a_matrix[1];
        const double a13 = a_matrix[2];
        const double a21 = a_matrix[3];
        const double a22 = a_matrix[4];
        const double a23 = a_matrix[5];
        const double a31 = a_matrix[6];
        const double a32 = a_matrix[7];
        const double a33 = a_matrix[8];

        // For a 3x3 matrix, the 3 invariants are the -trace, the sum of
        // principal minors, and the -determinant
        const double trace = a11 + a22 + a33;
        const double trace2 = (a11 * a11 + a12 * a21 + a13 * a31) +
                              (a12 * a21 + a22 * a22 + a23 * a32) +
                              (a13 * a31 + a23 * a32 + a33 * a33);
        const double det = a11 * (a22 * a33 - a23 * a32) -
                           a12 * (a21 * a33 - a23 * a31) +
                           a13 * (a21 * a32 - a22 * a31);

        std::complex<double> A(1.0, 0.0);
        const double Ar = 1.0;
        std::complex<double> B(-trace, 0.0);
        const double Br = -trace;
        std::complex<double> C(-0.5 * (trace2 - trace * trace), 0.0);
        const double Cr = -0.5 * (trace2 - trace * trace);
        std::complex<double> D(-det, 0.0);
        const double Dr = -det;
        const double Discrim = 18.0 * Ar * Br * Cr * Dr -
                               4.0 * Br * Br * Br * Dr + Br * Br * Cr * Cr -
                               4.0 * Ar * Cr * Cr * Cr -
                               27.0 * Ar * Ar * Dr * Dr;
        // Check whether real or complex eigenvalues
        if (Discrim >= 0) {
          // Equation has either 3 distinct real roots or a multiple root and
          // all roots are real lambda_ci not applicable
          LambdaCI_[k] = 0.0;
        } else {
          // Equation has one real root and two complex conjugate roots
          std::complex<double> Q;
          Q = std::sqrt(
            std::pow(
              B * B * B * 2.0 - A * B * C * 9.0 + A * A * D * 27.0, 2.0) -
            4.0 * std::pow(B * B - A * C * 3.0, 3.0));
          std::complex<double> CC;
          CC = std::pow(
            0.5 * (Q + 2.0 * B * B * B - 9.0 * A * B * C + 27.0 * A * A * D),
            1.0 / 3.0);
          if (Br * Br - 3.0 * Ar * Cr == 0.0) {
            Q = -Q;
            CC = std::pow(
              0.5 * (Q + 2.0 * B * B * B - 9.0 * A * B * C + 27.0 * A * A * D),
              1.0 / 3.0);
          }
          std::complex<double> II(0.0, -1.0);
          std::complex<double> EIG1;
          EIG1 = -B / (3.0 * A) - CC / (3.0 * A) -
                 (B * B - 3.0 * A * C) / (3.0 * A * CC);
          std::complex<double> EIG2;
          EIG2 = -B / (3.0 * A) + CC * (1.0 + II * std::sqrt(3.0)) / (6.0 * A) +
                 (1.0 - II * std::sqrt(3.0)) * (B * B - 3.0 * A * C) /
                   (6.0 * A * CC);
          std::complex<double> EIG3;
          EIG3 = -B / (3.0 * A) + CC * (1.0 - II * std::sqrt(3.0)) / (6.0 * A) +
                 (1.0 + II * std::sqrt(3.0)) * (B * B - 3.0 * A * C) /
                   (6.0 * A * CC);

          double maxEIG12 = std::max(std::imag(EIG1), std::imag(EIG2));
          LambdaCI_[k] = std::max(maxEIG12, std::imag(EIG3));
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_mean_resolved_ke ----------------------------------------
//--------------------------------------------------------------------------
void
TurbulenceAveragingPostProcessing::compute_mean_resolved_ke(
  const std::string& /* averageBlockName */, stk::mesh::Selector s_all_nodes)
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;

  const int ndim = realm_.spatialDimension_;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();

  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  const auto dualVol = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");

  nalu_ngp::ArrayDbl2 l_sum;
  Kokkos::Sum<nalu_ngp::ArrayDbl2> sum_reducer(l_sum);
  nalu_ngp::run_entity_par_reduce(
    "TurbPP::mean_res_tke", ngpMesh, stk::topology::NODE_RANK, s_all_nodes,
    KOKKOS_LAMBDA(const MeshIndex& mi, nalu_ngp::ArrayDbl2& pSum) {
      pSum.array_[0] += dualVol.get(mi, 0);

      double ke = 0.0;
      for (int d = 0; d < ndim; ++d)
        ke += velocity.get(mi, d) * velocity.get(mi, d);
      pSum.array_[1] += ke * dualVol.get(mi, 0) * 0.5;
    },
    sum_reducer);

  double g_sum[2] = {0.0, 0.0};
  auto comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_sum(comm, l_sum.array_, g_sum, 2);

  NaluEnv::self().naluOutputP0()
    << "Integrated ke and volume at time: " << g_sum[1] / g_sum[0] << " "
    << g_sum[0] << " " << realm_.get_current_time() << std::endl;
}

} // namespace nalu
} // namespace sierra
