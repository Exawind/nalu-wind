// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <SolutionOptions.h>
#include <Enums.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <FixPressureAtNodeInfo.h>

// basic c++
#include <stdexcept>
#include <utility>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// SolutionOptions - holder for user options at the realm scope
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SolutionOptions::SolutionOptions()
  : hybridDefault_(0.0),
    alphaDefault_(0.0),
    alphaUpwDefault_(1.0),
    upwDefault_(1.0),
    lamScDefault_(1.0),
    turbScDefault_(1.0),
    turbPrDefault_(1.0),
    nocDefault_(true),
    shiftedGradOpDefault_(false),
    skewSymmetricDefault_(false),
    tanhFormDefault_("classic"),
    tanhTransDefault_(2.0),
    tanhWidthDefault_(4.0),
    referenceDensity_(0.0),
    referenceTemperature_(298.0),
    thermalExpansionCoeff_(1.0),
    stefanBoltzmann_(5.6704e-8),
    nearestFaceEntrain_(0.0),
    includeDivU_(0.0),
    mdotInterpRhoUTogether_(true),
    solveIncompressibleContinuity_(false),
    isTurbulent_(false),
    turbulenceModel_(TurbulenceModel::LAMINAR),
    meshMotion_(false),
    meshTransformation_(false),
    externalMeshDeformation_(false),
    openfastFSI_(false),
    ncAlgGaussLabatto_(true),
    ncAlgUpwindAdvection_(true),
    ncAlgIncludePstab_(true),
    ncAlgDetailedOutput_(false),
    ncAlgCoincidentNodesErrorCheck_(false),
    ncAlgCurrentNormal_(false),
    ncAlgPngPenalty_(true),
    cvfemShiftMdot_(false),
    cvfemReducedSensPoisson_(false),
    inputVariablesRestorationTime_(1.0e8),
    inputVariablesInterpolateInTime_(false),
    inputVariablesPeriodicTime_(0.0),
    consistentMMPngDefault_(false),
    useConsolidatedSolverAlg_(false),
    useConsolidatedBcSolverAlg_(false),
    eigenvaluePerturb_(false),
    eigenvaluePerturbDelta_(0.0),
    eigenvaluePerturbBiasTowards_(3),
    eigenvaluePerturbTurbKe_(0.0),
    earthAngularVelocity_(7.2921159e-5),
    latitude_(0.0),
    raBoussinesqTimeScale_(-1.0),
    symmetryBcPenaltyFactor_(0.0),
    useStreletsUpwinding_(false),
    activateOpenMdotCorrection_(false),
    mdotAlgOpenCorrection_(0.0),
    explicitlyZeroOpenPressureGradient_(false),
    resetAMSAverages_(true),
    transition_model_(false),
    gammaEqActive_(false),
    lengthScaleLimiter_(false),
    referenceVelocity_(6.6),
    roughnessHeight_(0.1),
    RANSBelowKs_(false)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SolutionOptions::~SolutionOptions() {}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
SolutionOptions::load(const YAML::Node& y_node)
{
  const bool optional = true;
  const bool required = !optional;

  const YAML::Node y_solution_options =
    expect_map(y_node, "solution_options", optional);
  if (y_solution_options) {
    get_required(y_solution_options, "name", name_);
    get_if_present(
      y_solution_options, "nearest_face_entrainment", nearestFaceEntrain_,
      nearestFaceEntrain_);

    // penalty factor to apply to weak symmetry bc default to 2.0
    // Strengthening this penalty will better approximate the
    // boundary condition but will also increase the condition number of the
    // linear system and increase the error in the interior
    get_if_present(
      y_node, "symmetry_bc_penalty_factor", symmetryBcPenaltyFactor_);

    // divU factor for stress
    get_if_present(
      y_solution_options, "divU_stress_scaling", includeDivU_, includeDivU_);

    // mdot interpolation procedure
    get_if_present(
      y_solution_options, "interp_rhou_together_for_mdot",
      mdotInterpRhoUTogether_, mdotInterpRhoUTogether_);

    get_if_present(
      y_solution_options, "use_balanced_buoyancy_force",
      use_balanced_buoyancy_force_, use_balanced_buoyancy_force_);

    get_if_present(
      y_solution_options, "vof_sharpening_scaling_factor",
      vof_sharpening_scaling_factor_, vof_sharpening_scaling_factor_);

    get_if_present(
      y_solution_options, "vof_diffusion_scaling_factor",
      vof_diffusion_scaling_factor_, vof_diffusion_scaling_factor_);

    // Solve for incompressible continuity
    get_if_present(
      y_solution_options, "solve_incompressible_continuity",
      solveIncompressibleContinuity_, solveIncompressibleContinuity_);

    // external mesh motion expected
    get_if_present(
      y_solution_options, "externally_provided_mesh_deformation",
      externalMeshDeformation_, externalMeshDeformation_);

    // shift mdot for continuity (CVFEM)
    get_if_present(
      y_solution_options, "shift_cvfem_mdot", cvfemShiftMdot_, cvfemShiftMdot_);

    // DEPRECATED shifted CVFEM pressure poisson;
    if (y_solution_options["shift_cvfem_poisson"]) {
      throw std::runtime_error(
        "shift_cvfem_poisson line command is deprecated to the generalized "
        "solution options block command, shifted_gradient_operator");
    }

    // Reduce sensitivities for CVFEM PPE: M delta_p = b - A p; M has quadrature
    // at edge midpoints of the element
    get_if_present(
      y_solution_options, "reduced_sens_cvfem_poisson",
      cvfemReducedSensPoisson_, cvfemReducedSensPoisson_);

    // check for consolidated solver alg (AssembleSolver)
    get_if_present(
      y_solution_options, "use_consolidated_solver_algorithm",
      useConsolidatedSolverAlg_, useConsolidatedSolverAlg_);

    // check for consolidated face-elem bc alg
    get_if_present(
      y_solution_options, "use_consolidated_face_elem_bc_algorithm",
      useConsolidatedBcSolverAlg_, useConsolidatedBcSolverAlg_);

    // eigenvalue purturbation; over all dofs...
    get_if_present(
      y_solution_options, "eigenvalue_perturbation", eigenvaluePerturb_);
    get_if_present(
      y_solution_options, "eigenvalue_perturbation_delta",
      eigenvaluePerturbDelta_);
    get_if_present(
      y_solution_options, "eigenvalue_perturbation_bias_towards",
      eigenvaluePerturbBiasTowards_);
    get_if_present(
      y_solution_options, "eigenvalue_perturbation_turbulent_ke",
      eigenvaluePerturbTurbKe_);

    std::string projected_timescale_type = "default";
    get_if_present(
      y_solution_options, "projected_timescale_type", projected_timescale_type,
      projected_timescale_type);
    if (projected_timescale_type == "default")
      tscaleType_ = TSCALE_DEFAULT;
    else if (projected_timescale_type == "momentum_diag_inv")
      tscaleType_ = TSCALE_UDIAGINV;
    else
      throw std::runtime_error(
        "SolutionOptions: Invalid option provided for "
        "projected_timescale_type");

    // reset running AMS averages to instantaneous quantities during
    // intialization you would want to do this when restarting from a RANS
    // simulation
    get_if_present(
      y_solution_options, "reset_AMS_averages_on_init", resetAMSAverages_,
      resetAMSAverages_);

    // extract turbulence model; would be nice if we could parse an enum..
    std::string specifiedTurbModel;
    std::string defaultTurbModel = "laminar";
    get_if_present(
      y_solution_options, "turbulence_model", specifiedTurbModel,
      defaultTurbModel);

    bool matchedTurbulenceModel = false;
    for (int k = 0; k < static_cast<int>(TurbulenceModel::TurbulenceModel_END);
         ++k) {
      if (case_insensitive_compare(
            specifiedTurbModel, TurbulenceModelNames[k])) {
        turbulenceModel_ = TurbulenceModel(k);
        matchedTurbulenceModel = true;
        break;
      }
    }

    if (!matchedTurbulenceModel) {
      std::string msg =
        "Turbulence model `" + specifiedTurbModel +
        "' not implemented.\n  Available turbulence models are ";

      for (int k = 0;
           k < static_cast<int>(TurbulenceModel::TurbulenceModel_END); ++k) {
        msg += "`" + TurbulenceModelNames[k] + "'";
        if (k != static_cast<int>(TurbulenceModel::TurbulenceModel_END) - 1) {
          msg += ", ";
        }
      }
      throw std::runtime_error(msg);
    }
    if (turbulenceModel_ != TurbulenceModel::LAMINAR) {
      isTurbulent_ = true;
    }
    if (turbulenceModel_ == TurbulenceModel::SST_IDDES) {
      get_if_present(
        y_solution_options, "strelets_upwinding", useStreletsUpwinding_,
        useStreletsUpwinding_);
    }
    if (
      turbulenceModel_ == TurbulenceModel::SST ||
      turbulenceModel_ == TurbulenceModel::SST_IDDES) {
      get_if_present(
        y_solution_options, "transition_model", transition_model_,
        transition_model_);
      if (transition_model_ == true)
        gammaEqActive_ = true;
    }
    // initialize turbulence constants since some laminar models may need such
    // variables, e.g., kappa
    initialize_turbulence_constants();

    // extract possible copy from input fields restoration time
    get_if_present(
      y_solution_options, "input_variables_from_file_restoration_time",
      inputVariablesRestorationTime_, inputVariablesRestorationTime_);

    // choice of interpolation or snapping to closest in the data base
    get_if_present(
      y_solution_options, "input_variables_interpolate_in_time",
      inputVariablesInterpolateInTime_, inputVariablesInterpolateInTime_);

    // allow for periodic sampling in time
    get_if_present(
      y_solution_options, "input_variables_from_file_periodic_time",
      inputVariablesPeriodicTime_, inputVariablesPeriodicTime_);

    // check for global correction algorithm
    get_if_present(
      y_solution_options, "activate_open_mdot_correction",
      activateOpenMdotCorrection_, activateOpenMdotCorrection_);

    get_if_present(
      y_solution_options, "explicitly_zero_open_pressure_gradient",
      explicitlyZeroOpenPressureGradient_, explicitlyZeroOpenPressureGradient_);

    // first set of options; hybrid, source, etc.
    const YAML::Node y_options =
      expect_sequence(y_solution_options, "options", required);
    if (y_options) {
      for (size_t ioption = 0; ioption < y_options.size(); ++ioption) {
        const YAML::Node y_option = y_options[ioption];
        if (expect_map(y_option, "hybrid_factor", optional)) {
          y_option["hybrid_factor"] >> hybridMap_;
        } else if (expect_map(y_option, "alpha", optional)) {
          y_option["alpha"] >> alphaMap_;
        } else if (expect_map(y_option, "alpha_upw", optional)) {
          y_option["alpha_upw"] >> alphaUpwMap_;
        } else if (expect_map(y_option, "upw_factor", optional)) {
          y_option["upw_factor"] >> upwMap_;
        } else if (expect_map(y_option, "relaxation_factor", optional)) {
          y_option["relaxation_factor"] >> relaxFactorMap_;
        } else if (expect_map(y_option, "limiter", optional)) {
          y_option["limiter"] >> limiterMap_;
        } else if (expect_map(y_option, "laminar_schmidt", optional)) {
          y_option["laminar_schmidt"] >> lamScMap_;
        } else if (expect_map(y_option, "laminar_prandtl", optional)) {
          y_option["laminar_prandtl"] >> lamPrMap_;
        } else if (expect_map(y_option, "turbulent_schmidt", optional)) {
          y_option["turbulent_schmidt"] >> turbScMap_;
        } else if (expect_map(y_option, "turbulent_prandtl", optional)) {
          y_option["turbulent_prandtl"] >> turbPrMap_;
        } else if (expect_map(y_option, "source_terms", optional)) {
          const YAML::Node ySrc = y_option["source_terms"];
          ySrc >> srcTermsMap_;
        } else if (expect_map(y_option, "element_source_terms", optional)) {
          const YAML::Node ySrc = y_option["element_source_terms"];
          ySrc >> elemSrcTermsMap_;
        } else if (expect_map(y_option, "source_term_parameters", optional)) {
          y_option["source_term_parameters"] >> srcTermParamMap_;
        } else if (expect_map(
                     y_option, "element_source_term_parameters", optional)) {
          y_option["element_source_term_parameters"] >> elemSrcTermParamMap_;
        } else if (expect_map(y_option, "projected_nodal_gradient", optional)) {
          y_option["projected_nodal_gradient"] >> nodalGradMap_;
        } else if (expect_map(y_option, "noc_correction", optional)) {
          y_option["noc_correction"] >> nocMap_;
        } else if (expect_map(
                     y_option, "shifted_gradient_operator", optional)) {
          y_option["shifted_gradient_operator"] >> shiftedGradOpMap_;
        } else if (expect_map(y_option, "skew_symmetric_advection", optional)) {
          y_option["skew_symmetric_advection"] >> skewSymmetricMap_;
        } else if (expect_map(
                     y_option, "input_variables_from_file", optional)) {
          y_option["input_variables_from_file"] >> inputVarFromFileMap_;
        } else if (expect_map(
                     y_option, "turbulence_model_constants", optional)) {
          std::map<std::string, double> turbConstMap;
          y_option["turbulence_model_constants"] >> turbConstMap;
          // iterate the parsed map
          std::map<std::string, double>::iterator it;
          for (it = turbConstMap.begin(); it != turbConstMap.end(); ++it) {
            std::string theConstName = it->first;
            double theConstValue = it->second;
            // find the enum and set the value
            bool foundIt = false;
            for (int k = 0; k < TM_END; ++k) {
              if (theConstName == TurbulenceModelConstantNames[k]) {
                TurbulenceModelConstant theConstEnum =
                  TurbulenceModelConstant(k);
                turbModelConstantMap_[theConstEnum] = theConstValue;
                foundIt = true;
                break;
              }
            }
            // error check..
            if (!foundIt) {
              NaluEnv::self().naluOutputP0()
                << "Sorry, turbulence model constant with name " << theConstName
                << " was not found " << std::endl;
              NaluEnv::self().naluOutputP0()
                << "List of turbulence model constant names are as follows:"
                << std::endl;
              for (int k = 0; k < TM_END; ++k) {
                NaluEnv::self().naluOutputP0()
                  << TurbulenceModelConstantNames[k] << std::endl;
              }
            }
          }
        } else if (expect_map(y_option, "user_constants", optional)) {
          const YAML::Node y_user_constants = y_option["user_constants"];
          get_if_present(
            y_user_constants, "reference_density", referenceDensity_,
            referenceDensity_);
          get_if_present(
            y_user_constants, "reference_temperature", referenceTemperature_,
            referenceTemperature_);

          const auto thermal_expansion_option = "thermal_expansion_coefficient";
          if (
            y_user_constants["reference_temperature"] &&
            !y_user_constants[thermal_expansion_option]) {
            thermalExpansionCoeff_ = 1 / referenceTemperature_;
            NaluEnv::self().naluOutputP0()
              << "Using ideal gas relationship for thermal expansion "
                 "coefficient of "
              << thermalExpansionCoeff_ << "\n  -- specify "
              << thermal_expansion_option
              << " in user constants to set a different value" << std::endl;
          }
          get_if_present(
            y_user_constants, thermal_expansion_option, thermalExpansionCoeff_,
            thermalExpansionCoeff_);

          get_if_present(
            y_user_constants, "earth_angular_velocity", earthAngularVelocity_,
            earthAngularVelocity_);
          get_if_present(y_user_constants, "latitude", latitude_, latitude_);
          get_if_present(
            y_user_constants, "boussinesq_time_scale", raBoussinesqTimeScale_,
            raBoussinesqTimeScale_);
          get_if_present(
            y_user_constants, "roughness_height", roughnessHeight_,
            roughnessHeight_);
          get_if_present(
            y_user_constants, "length_scale_limiter", lengthScaleLimiter_,
            lengthScaleLimiter_);
          get_if_present(
            y_user_constants, "reference_velocity", referenceVelocity_,
            referenceVelocity_);
          get_if_present(
            y_user_constants, "rans_below_ks", RANSBelowKs_, RANSBelowKs_);

          if (expect_sequence(y_user_constants, "gravity", optional)) {
            const int gravSize = y_user_constants["gravity"].size();
            gravity_.resize(gravSize);
            for (int i = 0; i < gravSize; ++i) {
              gravity_[i] = y_user_constants["gravity"][i].as<double>();
            }
          }
          if (expect_sequence(y_user_constants, "body_force", optional)) {
            const int bodyForceSize = y_user_constants["body_force"].size();
            bodyForce_.resize(bodyForceSize);
            for (int i = 0; i < bodyForceSize; ++i) {
              bodyForce_[i] = y_user_constants["body_force"][i].as<double>();
            }
          }
          if (expect_sequence(y_user_constants, "east_vector", optional)) {
            const int vecSize = y_user_constants["east_vector"].size();
            eastVector_.resize(vecSize);
            for (int i = 0; i < vecSize; ++i) {
              eastVector_[i] = y_user_constants["east_vector"][i].as<double>();
              // y_user_constants["east_vector"][i] >> eastVector_[i];
            }
          }
          if (expect_sequence(y_user_constants, "north_vector", optional)) {
            const int vecSize = y_user_constants["north_vector"].size();
            northVector_.resize(vecSize);
            for (int i = 0; i < vecSize; ++i) {
              northVector_[i] =
                y_user_constants["north_vector"][i].as<double>();
              // y_user_constants["north_vector"][i] >> northVector_[i];
            }
          }
        } else if (expect_map(y_option, "non_conformal", optional)) {
          const YAML::Node y_nc = y_option["non_conformal"];
          get_if_present(
            y_nc, "gauss_labatto_quadrature", ncAlgGaussLabatto_,
            ncAlgGaussLabatto_);
          get_if_present(
            y_nc, "upwind_advection", ncAlgUpwindAdvection_,
            ncAlgUpwindAdvection_);
          get_if_present(
            y_nc, "include_pstab", ncAlgIncludePstab_, ncAlgIncludePstab_);
          get_if_present(
            y_nc, "detailed_output", ncAlgDetailedOutput_,
            ncAlgDetailedOutput_);
          get_if_present(
            y_nc, "activate_coincident_node_error_check",
            ncAlgCoincidentNodesErrorCheck_, ncAlgCoincidentNodesErrorCheck_);
          get_if_present(
            y_nc, "current_normal", ncAlgCurrentNormal_, ncAlgCurrentNormal_);
          get_if_present(
            y_nc, "include_png_penalty", ncAlgPngPenalty_, ncAlgPngPenalty_);
        } else if (expect_map(y_option, "peclet_function_form", optional)) {
          y_option["peclet_function_form"] >> tanhFormMap_;
        } else if (expect_map(
                     y_option, "peclet_function_tanh_transition", optional)) {
          y_option["peclet_function_tanh_transition"] >> tanhTransMap_;
        } else if (expect_map(
                     y_option, "peclet_function_tanh_width", optional)) {
          y_option["peclet_function_tanh_width"] >> tanhWidthMap_;
        }
        // overload line command, however, push to the same tanh data structure
        else if (expect_map(y_option, "blending_function_form", optional)) {
          y_option["blending_function_form"] >> tanhFormMap_;
        } else if (expect_map(y_option, "tanh_transition", optional)) {
          y_option["tanh_transition"] >> tanhTransMap_;
        } else if (expect_map(y_option, "tanh_width", optional)) {
          y_option["tanh_width"] >> tanhWidthMap_;
        } else if (expect_map(
                     y_option, "consistent_mass_matrix_png", optional)) {
          y_option["consistent_mass_matrix_png"] >> consistentMassMatrixPngMap_;
        } else if (expect_map(
                     y_option, "dynamic_body_force_box_parameters", optional)) {
          const YAML::Node yDyn = y_option["dynamic_body_force_box_parameters"];
          get_required(yDyn, "forcing_direction", dynamicBodyForceDir_);
          get_required(
            yDyn, "velocity_reference", dynamicBodyForceVelReference_);
          get_required(
            yDyn, "density_reference", dynamicBodyForceDenReference_);
          get_required(
            yDyn, "velocity_target_name", dynamicBodyForceVelTarget_);
          const int dragTargetSize = yDyn["drag_target_name"].size();
          dynamicBodyForceDragTarget_.resize(dragTargetSize);
          for (int i = 0; i < dragTargetSize; ++i) {
            dynamicBodyForceDragTarget_[i] =
              yDyn["drag_target_name"][i].as<std::string>();
          }
          get_required(yDyn, "output_file_name", dynamicBodyForceOutFile_);
          dynamicBodyForceBox_ = true;
        } else {
          if (!NaluEnv::self().parallel_rank()) {
            std::cout
              << "Error: parsing at "
              << NaluParsingHelper::info(y_option)
              //<< "... at parent ... " << NaluParsingHelper::info(y_node)
              << std::endl;
          }
          throw std::runtime_error(
            "unknown solution option: " + NaluParsingHelper::info(y_option));
        }
      }
    }

    // Handle old mesh motion section and throw an error early if the user is
    // attempting to use an old file with the latest branch
    if (y_solution_options["mesh_motion"]) {
      NaluEnv::self().naluOutput() << "SolutionOptions: Detected mesh motion "
                                      "section within solution_options. "
                                      "This is no longer supported. Please "
                                      "update your input file appropriately"
                                   << std::endl;
      throw std::runtime_error(
        "mesh_motion in solution_options is deprecated.");
    }

    const YAML::Node fix_pressure =
      expect_map(y_solution_options, "fix_pressure_at_node", optional);
    if (fix_pressure) {
      needPressureReference_ = true;
      fixPressureInfo_.reset(new FixPressureAtNodeInfo);

      fixPressureInfo_->refPressure_ = fix_pressure["value"].as<double>();
      if (fix_pressure["node_lookup_type"]) {
        std::string lookupTypeName =
          fix_pressure["node_lookup_type"].as<std::string>();
        if (lookupTypeName == "stk_node_id")
          fixPressureInfo_->lookupType_ = FixPressureAtNodeInfo::STK_NODE_ID;
        else if (lookupTypeName == "spatial_location")
          fixPressureInfo_->lookupType_ =
            FixPressureAtNodeInfo::SPATIAL_LOCATION;
        else
          throw std::runtime_error(
            "FixPressureAtNodeInfo: Invalid option specified for "
            "'node_lookup_type' in input file.");
      }

      if (
        fixPressureInfo_->lookupType_ ==
        FixPressureAtNodeInfo::SPATIAL_LOCATION) {
        fixPressureInfo_->location_ =
          fix_pressure["location"].as<std::vector<double>>();
        fixPressureInfo_->searchParts_ =
          fix_pressure["search_target_part"].as<std::vector<std::string>>();
        if (fix_pressure["search_method"]) {
          std::string searchMethodName =
            fix_pressure["search_method"].as<std::string>();
          if (searchMethodName == "boost_rtree") {
            fixPressureInfo_->searchMethod_ = stk::search::KDTREE;
            NaluEnv::self().naluOutputP0()
              << "Warning: search method 'boost_rtree' has been"
              << " deprecated. Switching to 'stk_kdtree'." << std::endl;
          } else if (searchMethodName == "stk_kdtree")
            fixPressureInfo_->searchMethod_ = stk::search::KDTREE;
          else
            NaluEnv::self().naluOutputP0()
              << "ABL Fix Pressure: Search will use stk_kdtree" << std::endl;
        }
      } else {
        fixPressureInfo_->stkNodeId_ =
          fix_pressure["node_identifier"].as<unsigned int>();
      }
    }
  }

  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "Turbulence Model Review:   " << std::endl;
  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Turbulence Model is: "
    << TurbulenceModelNames[static_cast<int>(turbulenceModel_)] << " "
    << isTurbulent_ << std::endl;
  if (gammaEqActive_ == true) {
    if (turbModelConstantMap_[TM_fsti] > 0) {
      NaluEnv::self().naluOutputP0()
        << "Transition Model is: One Equation Gamma w/ constant Tu"
        << std::endl;
    } else {
      NaluEnv::self().naluOutputP0()
        << "Transition Model is: One Equation Gamma w/ local Tu" << std::endl;
    }
  } else {
    NaluEnv::self().naluOutputP0() << "No Transition Model" << std::endl;
  }

  // over view PPE specifications
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "PPE review:   " << std::endl;
  NaluEnv::self().naluOutputP0() << "===========================" << std::endl;

  if (cvfemShiftMdot_)
    NaluEnv::self().naluOutputP0()
      << "Shifted CVFEM mass flow rate" << std::endl;
  if (cvfemReducedSensPoisson_)
    NaluEnv::self().naluOutputP0()
      << "Reduced sensitivities CVFEM Poisson" << std::endl;

  // sanity checks; if user asked for shifted Poisson, then user will have
  // reduced sensitivities
  if (get_shifted_grad_op("pressure")) {
    if (!cvfemReducedSensPoisson_) {
      NaluEnv::self().naluOutputP0()
        << "Reduced sensitivities CVFEM Poisson will be set since reduced "
           "grad_op is requested"
        << std::endl;
      cvfemReducedSensPoisson_ = true;
    }
  }

  // overview gradient operator for CVFEM
  if (shiftedGradOpMap_.size() > 0) {
    NaluEnv::self().naluOutputP0() << std::endl;
    NaluEnv::self().naluOutputP0()
      << "CVFEM gradient operator review:   " << std::endl;
    NaluEnv::self().naluOutputP0()
      << "===========================" << std::endl;
    for (const auto& shiftIt : shiftedGradOpMap_) {
      NaluEnv::self().naluOutputP0()
        << " dof: " << shiftIt.first
        << " shifted: " << (shiftIt.second ? "yes" : "no") << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- initialize_turbulence_constants ---------------------------------
//--------------------------------------------------------------------------
void
SolutionOptions::initialize_turbulence_constants()
{
  // set the default map values; resize to max turbulence model enum
  turbModelConstantMap_[TM_cMu] = 0.09;
  turbModelConstantMap_[TM_kappa] = 0.41;
  turbModelConstantMap_[TM_cDESke] = 0.61;
  turbModelConstantMap_[TM_cDESkw] = 0.78;
  turbModelConstantMap_[TM_tkeProdLimitRatio] =
    (turbulenceModel_ == TurbulenceModel::SST ||
     turbulenceModel_ == TurbulenceModel::SSTLR ||
     turbulenceModel_ == TurbulenceModel::SST_DES ||
     turbulenceModel_ == TurbulenceModel::SST_AMS ||
     turbulenceModel_ == TurbulenceModel::SST_IDDES)
      ? 10.0
      : 500.0;
  turbModelConstantMap_[TM_cmuEps] = 0.0856;
  turbModelConstantMap_[TM_cEps] = 0.845;
  turbModelConstantMap_[TM_betaStar] = 0.09;
  turbModelConstantMap_[TM_aOne] = 0.31;
  turbModelConstantMap_[TM_betaOne] = 0.075;
  turbModelConstantMap_[TM_betaTwo] = 0.0828;
  turbModelConstantMap_[TM_gammaOne] = 5.0 / 9.0;
  turbModelConstantMap_[TM_gammaTwo] = 0.44;
  turbModelConstantMap_[TM_sigmaKOne] = 0.85;
  turbModelConstantMap_[TM_sigmaKTwo] = 1.0;
  turbModelConstantMap_[TM_sigmaWOne] = 0.50;
  turbModelConstantMap_[TM_sigmaWTwo] = 0.856;
  turbModelConstantMap_[TM_cmuCs] = 0.17;
  turbModelConstantMap_[TM_Cw] = 0.325;
  turbModelConstantMap_[TM_CbTwo] = 0.35;
  turbModelConstantMap_[TM_SDRWallFactor] = 10.0;
  turbModelConstantMap_[TM_zCV] = 0.5;
  turbModelConstantMap_[TM_ci] = 0.9;
  turbModelConstantMap_[TM_elog] = 9.8;
  turbModelConstantMap_[TM_yplus_crit] = 11.63;
  turbModelConstantMap_[TM_CMdeg] = 0.11;
  turbModelConstantMap_[TM_forCl] = 4.0;
  turbModelConstantMap_[TM_forCeta] = 70.0;
  turbModelConstantMap_[TM_forCt] = 6.0;
  turbModelConstantMap_[TM_forBlT] = 1.0;
  turbModelConstantMap_[TM_forBlKol] = 1.0;
  turbModelConstantMap_[TM_forFac] = 8.0;
  turbModelConstantMap_[TM_v2cMu] = 0.22;
  turbModelConstantMap_[TM_aspRatSwitch] = 64.0;
  turbModelConstantMap_[TM_periodicForcingLengthX] = M_PI;
  turbModelConstantMap_[TM_periodicForcingLengthY] = 0.25;
  turbModelConstantMap_[TM_periodicForcingLengthZ] = 3.0 / 8.0 * M_PI;
  turbModelConstantMap_[TM_sigmaMax] = 1.0;
  turbModelConstantMap_[TM_ch1] = 3.0;
  turbModelConstantMap_[TM_ch2] = 1.0;
  turbModelConstantMap_[TM_ch3] = 0.5;
  turbModelConstantMap_[TM_tau_des] = 100.0 / 15.0;
  turbModelConstantMap_[TM_iddes_Cw] = 0.15;
  turbModelConstantMap_[TM_iddes_Cdt1] = 20.0;
  turbModelConstantMap_[TM_iddes_Cdt2] = 3.0;
  turbModelConstantMap_[TM_iddes_Cl] = 5.0;
  turbModelConstantMap_[TM_iddes_Ct] = 1.87;
  turbModelConstantMap_[TM_abl_bndtw] = 5.0;
  turbModelConstantMap_[TM_abl_deltandtw] = 1.0;
  turbModelConstantMap_[TM_abl_sigma] = 2.0;
  turbModelConstantMap_[TM_ams_peclet_offset] = 0.6;
  turbModelConstantMap_[TM_ams_peclet_slope] = 12.0;
  turbModelConstantMap_[TM_ams_peclet_scale] = 100.0;
  turbModelConstantMap_[TM_fMuExp] = -0.0115;
  turbModelConstantMap_[TM_utau] = 1.0;
  turbModelConstantMap_[TM_cEpsOne] = 1.35;
  turbModelConstantMap_[TM_cEpsTwo] = 1.80;
  turbModelConstantMap_[TM_fOne] = 1.0;
  turbModelConstantMap_[TM_sigmaK] = 1.0;
  turbModelConstantMap_[TM_sigmaEps] = 1.3;
  turbModelConstantMap_[TM_sstLRDestruct] = 1.0;
  turbModelConstantMap_[TM_sstLRProd] = 1.0;
  turbModelConstantMap_[TM_tkeAmb] = 0.0;
  turbModelConstantMap_[TM_sdrAmb] = 0.0;
  turbModelConstantMap_[TM_avgTimeCoeff] = 1.0;
  turbModelConstantMap_[TM_alphaInf] = 0.52;
  turbModelConstantMap_[TM_fsti] = -1;
}

double
SolutionOptions::get_alpha_factor(const std::string& dofName) const
{
  double factor = alphaDefault_;
  auto iter = alphaMap_.find(dofName);

  if (iter != alphaMap_.end())
    factor = iter->second;

  return factor;
}

double
SolutionOptions::get_alpha_upw_factor(const std::string& dofName) const
{
  double factor = alphaUpwDefault_;
  auto iter = alphaUpwMap_.find(dofName);

  if (iter != alphaUpwMap_.end())
    factor = iter->second;

  return factor;
}

double
SolutionOptions::get_upw_factor(const std::string& dofName) const
{
  double factor = upwDefault_;
  auto iter = upwMap_.find(dofName);

  if (iter != upwMap_.end())
    factor = iter->second;

  return factor;
}

double
SolutionOptions::get_relaxation_factor(const std::string& dofName) const
{
  double factor = relaxFactorDefault_;

  auto iter = relaxFactorMap_.find(dofName);
  if (iter != relaxFactorMap_.end())
    factor = iter->second;

  return factor;
}

bool
SolutionOptions::primitive_uses_limiter(const std::string& dofName) const
{
  bool usesIt = false;
  auto iter = limiterMap_.find(dofName);
  if (iter != limiterMap_.end())
    usesIt = iter->second;

  return usesIt;
}

bool
SolutionOptions::get_shifted_grad_op(const std::string& dofName) const
{
  bool factor = shiftedGradOpDefault_;
  auto iter = shiftedGradOpMap_.find(dofName);

  if (iter != shiftedGradOpMap_.end())
    factor = iter->second;

  return factor;
}

bool
SolutionOptions::get_skew_symmetric(const std::string& dofName) const
{
  bool factor = skewSymmetricDefault_;
  auto iter = skewSymmetricMap_.find(dofName);

  if (iter != skewSymmetricMap_.end())
    factor = iter->second;

  return factor;
}

std::vector<double>
SolutionOptions::get_gravity_vector(const unsigned nDim) const
{
  if (nDim != gravity_.size())
    throw std::runtime_error(
      "SolutionOptions::get_gravity_vector():Error "
      "Expected size does not equaly nDim");
  else
    return gravity_;
}

//--------------------------------------------------------------------------
//-------- get_turb_model_constant() ------------------------------------------
//--------------------------------------------------------------------------
double
SolutionOptions::get_turb_model_constant(
  TurbulenceModelConstant turbModelEnum) const
{
  std::map<TurbulenceModelConstant, double>::const_iterator it =
    turbModelConstantMap_.find(turbModelEnum);
  if (it != turbModelConstantMap_.end()) {
    return it->second;
  } else {
    throw std::runtime_error("unknown (not found) turbulence model constant");
  }
}

bool
SolutionOptions::get_noc_usage(const std::string& dofName) const
{
  bool factor = nocDefault_;
  std::map<std::string, bool>::const_iterator iter = nocMap_.find(dofName);
  if (iter != nocMap_.end()) {
    factor = (*iter).second;
  }
  return factor;
}

bool
SolutionOptions::has_set_boussinesq_time_scale()
{
  return (raBoussinesqTimeScale_ > std::numeric_limits<double>::min());
}

} // namespace nalu
} // namespace sierra
