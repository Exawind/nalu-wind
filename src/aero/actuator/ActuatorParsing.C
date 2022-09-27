// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorParsing.h>
#include <aero/actuator/ActuatorBulk.h>
#include <aero/actuator/ActuatorInfo.h>
#include <stdexcept>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

/**
 * @brief General options for each actuator instance
 *
 * @param actMeta
 * @param y_actuator
 */
void
actuator_instance_parse(ActuatorMeta& actMeta, const YAML::Node& y_actuator)
{
  actMeta.numNearestPointsFllcInt_.modify_host();

  for (int i = 0; i < actMeta.numberOfActuators_; i++) {
    std::string key;

    // TODO: I really don't like that we have these conditionals here.
    // I'd really like to align the naming conventions so the data in these
    // cases can be agnostic to actuator types
    switch (actMeta.actuatorType_) {
    case (ActuatorType::ActLineSimpleNGP): {
      key = "Blade" + std::to_string(i);
      break;
    }
    default: {
      key = "Turbine" + std::to_string(i);
      break;
    }
    }
    const YAML::Node y_instance = y_actuator[key];

    get_if_present_no_default(
      y_instance, "fllt_correction", actMeta.entityFLLC_(i));
    if (actMeta.entityFLLC_(i)) {
      actMeta.useFLLC_ = true;
    }

    get_required(
      y_instance, "num_force_pts_blade",
      actMeta.numNearestPointsFllcInt_.h_view(i));
    get_if_present_no_default(
      y_instance, "fllt_num_nearest_point",
      actMeta.numNearestPointsFllcInt_.h_view(i));
  }
} // namespace nalu

ActuatorType
get_backward_compatible_type(const std::string typeName)
{
  ActuatorType theParsedType = ActuatorTypeMap[typeName];
  switch (theParsedType) {
  case (ActuatorType::ActLineSimpleNGP):
  case (ActuatorType::ActLineSimple):
    return ActuatorType::ActLineSimpleNGP;
  case (ActuatorType::ActLineFAST):
  case (ActuatorType::ActLineFASTNGP):
    return ActuatorType::ActLineFASTNGP;
  case (ActuatorType::ActDiskFAST):
  case (ActuatorType::ActDiskFASTNGP):
    return ActuatorType::ActDiskFASTNGP;
  case (ActuatorType::ActLinePointDrag):
    return ActuatorType::ActLinePointDrag;
  default: {
    throw std::runtime_error("ActuatorType not supported in this context.");
  }
  }
}

/*! \brief Parse parameters to construct meta data for actuators
 *  Parse parameters and construct meta data for actuators.
 *  Intent is to divorce object creation/memory allocation from parsing
 *  to facilitate device compatibility.
 *
 *  This also has the added benefit of increasing unittest-ability.
 *
 *  General parameters that apply to all actuators should be parsed here.
 *  More specific actuator methods (i.e. LineFAST, DiskFAST) should implement
 *  another parse function that takes one YAML::Node and one ActuatorMeta object
 *  as inputs and returns a more specialized ActuatorMeta object.
 */
ActuatorMeta
actuator_parse(const YAML::Node& y_node)
{
  const YAML::Node y_actuator = y_node["actuator"];
  ThrowErrorMsgIf(
    !y_actuator, "actuator argument is "
                 "missing from yaml node passed to actuator_parse");
  int nTurbines = 0;
  std::string actuatorTypeName;
  ActuatorType actModelType;

  get_required(y_actuator, "type", actuatorTypeName);

  actModelType = get_backward_compatible_type(actuatorTypeName);

  if (actModelType == ActuatorType::ActLineSimpleNGP) {
    get_required(y_actuator, "n_simpleblades", nTurbines);
  } else {
    get_required(y_actuator, "n_turbines_glob", nTurbines);
  }

  ActuatorMeta actMeta(nTurbines, actModelType);
  // search specifications
  std::string searchMethodName = "na";
  get_if_present(
    y_actuator, "search_method", searchMethodName, searchMethodName);
  // determine search method for this pair
  if (searchMethodName == "boost_rtree") {
    actMeta.searchMethod_ = stk::search::KDTREE;
    NaluEnv::self().naluOutputP0()
      << "Warning: search method 'boost_rtree'"
      << " is being deprecated, switching to 'stk_kdtree'" << std::endl;
  } else if (searchMethodName == "stk_kdtree")
    actMeta.searchMethod_ = stk::search::KDTREE;
  else
    NaluEnv::self().naluOutputP0()
      << "Actuator::search method not declared; will use stk_kdtree"
      << std::endl;
  // extract the set of from target names; each spec is homogeneous in this
  // respect
  const YAML::Node searchTargets = y_actuator["search_target_part"];
  if (searchTargets) {
    if (searchTargets.Type() == YAML::NodeType::Scalar) {
      actMeta.searchTargetNames_.resize(1);
      actMeta.searchTargetNames_[0] = searchTargets.as<std::string>();
    } else {
      actMeta.searchTargetNames_.resize(searchTargets.size());
      for (size_t i = 0; i < searchTargets.size(); ++i) {
        actMeta.searchTargetNames_[i] = searchTargets[i].as<std::string>();
      }
    }
  } else {
    throw std::runtime_error("Actuator:: search_target_part is not declared.");
  }

  actuator_instance_parse(actMeta, y_actuator);

  return actMeta;
}

/**
 * @brief Standard interface for parsing epsilon values to be reused by sub
 * models
 *
 * @param iTurb
 * @param turbNode
 * @param actMeta
 */
void
epsilon_parsing(int iTurb, const YAML::Node& turbNode, ActuatorMeta& actMeta)
{
  // The value epsilon / chord [non-dimensional]
  // This is a vector containing the values for:
  //   - chord aligned (x),
  //   - tangential to chord (y),
  //   - spanwise (z)
  const YAML::Node epsilon_chord = turbNode["epsilon_chord"];
  const YAML::Node epsilon = turbNode["epsilon"];
  if (epsilon && epsilon_chord) {
    throw std::runtime_error(
      "epsilon and epsilon_chord have both been specified for Turbine " +
      std::to_string(iTurb) + "\nYou must pick one or the other.");
  }
  if (epsilon && actMeta.useFLLC_) {
    throw std::runtime_error(
      "epsilon and fllt_correction have both been specified for "
      "Turbine " +
      std::to_string(iTurb) +
      "\nepsilon_chord and epsilon_min should be used with "
      "fllt_correction.");
  }

  std::vector<double> epsilonTemp(3);
  if (epsilon) {
    // only require epsilon
    if (epsilon.Type() == YAML::NodeType::Scalar) {
      double isotropicEpsilon;
      get_required(turbNode, "epsilon", isotropicEpsilon);
      actMeta.isotropicGaussian_ = true;
      for (int j = 0; j < 3; j++) {
        actMeta.epsilon_.h_view(iTurb, j) = isotropicEpsilon;
      }
    } else {
      get_required(turbNode, "epsilon", epsilonTemp);
      for (int j = 0; j < 3; j++) {
        actMeta.epsilon_.h_view(iTurb, j) = epsilonTemp[j];
      }
      if (
        epsilonTemp[0] == epsilonTemp[1] && epsilonTemp[1] == epsilonTemp[2]) {
        actMeta.isotropicGaussian_ = true;
      }
    }
  } else if (epsilon_chord) {
    // require epsilon chord and epsilon min
    get_required(turbNode, "epsilon_chord", epsilonTemp);
    if (epsilonTemp[0] == epsilonTemp[1] && epsilonTemp[1] == epsilonTemp[2]) {
      actMeta.isotropicGaussian_ = true;
    }
    for (int j = 0; j < 3; j++) {
      if (epsilonTemp[j] <= 0.0) {
        throw std::runtime_error(
          "ERROR:: zero value for epsilon_chord detected. "
          "All epsilon components must be greater than zero");
      }
      actMeta.epsilonChord_.h_view(iTurb, j) = epsilonTemp[j];
    }

    // Minimum epsilon allowed in simulation. This is required when
    //   specifying epsilon/chord
    get_required(turbNode, "epsilon_min", epsilonTemp);
    if (!(actMeta.isotropicGaussian_ && epsilonTemp[0] == epsilonTemp[1] &&
          epsilonTemp[1] == epsilonTemp[2])) {
      actMeta.isotropicGaussian_ = false;
    }
    for (int j = 0; j < 3; j++) {
      actMeta.epsilon_.h_view(iTurb, j) = epsilonTemp[j];
    }
  } else {
    throw std::runtime_error(
      "Neither epsilon or epsilon_chord was declared in the input deck.  "
      "One of these two options must be used for the actuator model "
      "specified.");
  }
  // check epsilon values
  for (int j = 0; j < 3; j++) {
    if (actMeta.epsilon_.h_view(iTurb, j) <= 0.0) {
      throw std::runtime_error(
        "ERROR:: zero value for epsilon detected. "
        "All epsilon components must be greater than zero");
    }
  }
}
} // namespace nalu
} // namespace sierra
