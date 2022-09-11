// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NaluParsing_h
#define NaluParsing_h

#include <BoundaryConditions.h>
#include <Enums.h>
#include <InitialConditions.h>
#include <MaterialPropertys.h>
#include <NaluParsedTypes.h>
#include <NaluParsingHelper.h>
#include <NaluEnv.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace sierra {
namespace nalu {

// base class
struct UserData
{
  std::map<std::string, bool> bcDataSpecifiedMap_;
  std::map<std::string, UserDataType> bcDataTypeMap_;
  std::map<std::string, std::string> userFunctionMap_;
  std::map<std::string, std::vector<double>> functionParams_;
  std::map<std::string, std::vector<std::string>> functionStringParams_;

  // FIXME: must elevate temperature due to the temperature_bc_setup method
  Temperature temperature_;
  bool tempSpec_;
  bool externalData_;
  UserData() : tempSpec_(false), externalData_(false) {}
};

// packaged
struct WallUserData : public UserData
{
  Velocity u_;
  Velocity dx_;
  TurbKinEnergy tke_;
  TotDissRate tdr_;
  MixtureFraction mixFrac_;
  MassFraction massFraction_;
  NormalHeatFlux q_;
  ReferenceTemperature referenceTemperature_;
  Pressure pressure_;
  unsigned gravityComponent_;
  RoughnessHeight z0_;
  double uRef_;
  double zRef_;

  bool isAdiabatic_;
  bool isNoSlip_;
  bool heatFluxSpec_;
  bool isInterface_;
  bool refTempSpec_;

  bool RANSAblBcApproach_;
  bool wallFunctionApproach_;
  bool ablWallFunctionApproach_;
  YAML::Node ablWallFunctionNode_;

  bool isFsiInterface_;

  WallUserData()
    : UserData(),
      gravityComponent_(3),
      uRef_(6.6),
      zRef_(90.0),
      isAdiabatic_(false),
      isNoSlip_(false),
      heatFluxSpec_(false),
      isInterface_(false),
      refTempSpec_(false),
      RANSAblBcApproach_(false),
      wallFunctionApproach_(false),
      ablWallFunctionApproach_(false),
      isFsiInterface_(false)
  {
  }
};

struct InflowUserData : public UserData
{
  Velocity u_;
  TurbKinEnergy tke_;
  SpecDissRate sdr_;
  TotDissRate tdr_;
  MixtureFraction mixFrac_;
  MassFraction massFraction_;
  GammaInf gamma_;

  bool uSpec_;
  bool tkeSpec_;
  bool sdrSpec_;
  bool tdrSpec_;
  bool mixFracSpec_;
  bool massFractionSpec_;
  bool gammaSpec_;
  InflowUserData()
    : UserData(),
      uSpec_(false),
      tkeSpec_(false),
      sdrSpec_(false),
      tdrSpec_(false),
      mixFracSpec_(false),
      massFractionSpec_(false),
      gammaSpec_(false)
  {
  }
};

struct OpenUserData : public UserData
{
  Velocity u_;
  Pressure p_;
  TurbKinEnergy tke_;
  SpecDissRate sdr_;
  TotDissRate tdr_;
  MixtureFraction mixFrac_;
  MassFraction massFraction_;
  GammaOpen gamma_;

  bool uSpec_;
  bool pSpec_;
  bool tkeSpec_;
  bool sdrSpec_;
  bool tdrSpec_;
  bool mixFracSpec_;
  bool massFractionSpec_;
  bool totalP_;
  bool gammaSpec_;
  EntrainmentMethod entrainMethod_;

  OpenUserData()
    : UserData(),
      uSpec_(false),
      pSpec_(false),
      tkeSpec_(false),
      sdrSpec_(false),
      tdrSpec_(false),
      mixFracSpec_(false),
      massFractionSpec_(false),
      totalP_{false},
      gammaSpec_(false),
      entrainMethod_{EntrainmentMethod::COMPUTED}
  {
  }
};

struct OversetUserData : public UserData
{
  // at present, simulation can have one background mesh with multiple,
  // non-interacting overset blocks

  /// Percentage overlap between background and interior mesh
  double percentOverlap_;
  bool clipIsoParametricCoords_;
  bool detailedOutput_;
  /// Part name for the background  mesh
  std::string backgroundBlock_;

  /// Part name for the interior fringe surface created on the background mesh
  /// by hole cutting algorithm
  std::string backgroundSurface_;

  /// Part name for the inactive elements on the background mesh as a result of
  /// hole cutting.
  std::string backgroundCutBlock_;

  /// Exterior boundary of the internal meshe(s) that are mandatory receptors
  std::string oversetSurface_;

  /// List of part names for the interior meshes
  std::vector<std::string> oversetBlockVec_;

#ifdef NALU_USES_TIOGA
  YAML::Node oversetBlocks_;
#endif

  OversetUserData()
    : UserData(),
      percentOverlap_(10.0),
      clipIsoParametricCoords_(false),
      detailedOutput_(false),
      backgroundBlock_("na"),
      backgroundSurface_("na"),
      backgroundCutBlock_("na"),
      oversetSurface_("na")
  {
  }
};

struct SymmetryUserData : public UserData
{
  enum class SymmetryTypes {
    GENERAL_WEAK,
    X_DIR_STRONG,
    Y_DIR_STRONG,
    Z_DIR_STRONG
  };
  SymmetryTypes symmType_;
  bool useProjections_;
  SymmetryUserData()
    : UserData(), symmType_(SymmetryTypes::GENERAL_WEAK), useProjections_(false)
  {
  }
};

struct ABLTopUserData : public UserData
{
  NormalTemperatureGradient normalTemperatureGradient_;

  bool ABLTopBC_{false};
  std::vector<int> grid_dims_;
  std::vector<int> horiz_bcs_;
  double z_sample_;

  bool normalTemperatureGradientSpec_;

  ABLTopUserData()
    : UserData(), z_sample_(-999.0), normalTemperatureGradientSpec_(false)
  {
  }
};

struct PeriodicUserData : public UserData
{

  double searchTolerance_;
  std::string searchMethodName_;

  PeriodicUserData()
    : UserData(), searchTolerance_(1.0e-8), searchMethodName_("na")
  {
  }
};

struct NonConformalUserData : public UserData
{
  std::string searchMethodName_;
  double expandBoxPercentage_;
  bool clipIsoParametricCoords_;
  double searchTolerance_;
  bool dynamicSearchTolAlg_;
  NonConformalUserData()
    : UserData(),
      searchMethodName_("na"),
      expandBoxPercentage_(0.0),
      clipIsoParametricCoords_(false),
      searchTolerance_(1.0e-16),
      dynamicSearchTolAlg_(false)
  {
  }
};

struct WallBoundaryConditionData : public BoundaryCondition
{
  WallBoundaryConditionData(){};
  WallUserData userData_;
};

struct InflowBoundaryConditionData : public BoundaryCondition
{
  InflowBoundaryConditionData(){};
  InflowUserData userData_;
};

struct OpenBoundaryConditionData : public BoundaryCondition
{
  OpenBoundaryConditionData(){};
  OpenUserData userData_;
};

struct OversetBoundaryConditionData : public BoundaryCondition
{
  enum OversetAPI {
    TPL_TIOGA = 0,   ///< Overset connectivity using TIOGA
    OVERSET_NONE = 1 ///< Guard for error messages
  };

  OversetBoundaryConditionData(){};
  OversetUserData userData_;
  OversetAPI oversetConnectivityType_;
};

struct SymmetryBoundaryConditionData : public BoundaryCondition
{
  SymmetryBoundaryConditionData(){};
  SymmetryUserData userData_;
};

struct ABLTopBoundaryConditionData : public BoundaryCondition
{
  ABLTopBoundaryConditionData(){};
  ABLTopUserData userData_;
  SymmetryUserData symmetryUserData_;
};

struct PeriodicBoundaryConditionData : public BoundaryCondition
{
  PeriodicBoundaryConditionData(){};
  MasterSlave masterSlave_;
  PeriodicUserData userData_;
};

struct NonConformalBoundaryConditionData : public BoundaryCondition
{
  NonConformalBoundaryConditionData(){};
  std::vector<std::string> currentPartNameVec_;
  std::vector<std::string> opposingPartNameVec_;
  NonConformalUserData userData_;
};

struct BoundaryConditionOptions
{
  std::string bcSetName_;
  WallBoundaryConditionData wallbc_;
  InflowBoundaryConditionData inflowbc_;
  OpenBoundaryConditionData openbc_;
  OversetBoundaryConditionData oversetbc_;
  NonConformalBoundaryConditionData nonConformalbc_;
  SymmetryBoundaryConditionData symmetrybc_;
  ABLTopBoundaryConditionData abltopbc_;
  PeriodicBoundaryConditionData periodicbc_;
};

struct MeshInput
{
  std::string meshName_;
};

// initial conditions
struct ConstantInitialConditionData : public InitialCondition
{
  ConstantInitialConditionData(bool debug) : debug_(debug) {}
  std::vector<std::string> fieldNames_;
  std::vector<std::vector<double>> data_;
  const bool debug_;
};

struct UserFunctionInitialConditionData : public InitialCondition
{
  UserFunctionInitialConditionData() {}
  std::map<std::string, std::string> functionNames_;
  std::map<std::string, std::vector<double>> functionParams_;
};

inline bool
string_represents_positive_integer(std::string v)
{
  return !v.empty() && v.find_first_not_of("0123456789") == std::string::npos;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
get_yaml_value(const YAML::Node& v)
{
  // yaml will parse inputs with leading zeros as octals if
  // the number is an octal, e.g. max_itertions: 0010 is
  // equivalent to max_iterations: 8.
  // this works around that to have 0010 equivalent to 10
  if (string_represents_positive_integer(v.template as<std::string>())) {
    return std::stoi(v.template as<std::string>());
  } else {
    return v.template as<T>();
  }
}

template <typename T>
typename std::enable_if<!std::is_integral<T>::value, T>::type
get_yaml_value(const YAML::Node& v)
{
  return v.template as<T>();
}

/// Set @param result if the @param key is present in the @param node, else set
/// it to the given default value
template <typename T>
void
get_if_present(
  const YAML::Node& node,
  const std::string& key,
  T& result,
  const T& default_if_not_present = T())
{
  if (node[key]) {
    result = get_yaml_value<T>(node[key]);
  } else {
    result = default_if_not_present;
  }
}

/// this version doesn't change @param result unless the @param key is present
/// in the @param node
template <typename T>
void
get_if_present_no_default(
  const YAML::Node& node, const std::string& key, T& result)
{
  if (node[key]) {
    result = get_yaml_value<T>(node[key]);
  }
}

/// this version requires the @param key to be present
template <typename T>
void
get_required(const YAML::Node& node, const std::string& key, T& result)
{
  if (node[key]) {
    result = get_yaml_value<T>(node[key]);
  } else {
    if (!NaluEnv::self().parallel_rank()) {
      std::ostringstream err_msg;
      err_msg << "\n\nError: parsing missing required key: " << key << " at "
              << NaluParsingHelper::line_info(node)
              << " for node= " << std::endl;
      NaluParsingHelper::emit(err_msg, node);
      std::cout << err_msg.str() << std::endl;
    }
    throw std::runtime_error("Error: parsing missing required key: " + key);
  }
}

/// these can be used to check and ensure a type of yaml node is as expected
const YAML::Node expect_type(
  const YAML::Node& node,
  const std::string& key,
  YAML::NodeType::value type,
  bool optional = false);

const YAML::Node expect_null(
  const YAML::Node& node, const std::string& key, bool optional = false);

const YAML::Node expect_scalar(
  const YAML::Node& node, const std::string& key, bool optional = false);

const YAML::Node expect_sequence(
  const YAML::Node& node, const std::string& key, bool optional = false);

const YAML::Node expect_map(
  const YAML::Node& node, const std::string& key, bool optional = false);

void operator>>(const YAML::Node& node, WallBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, InflowBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, OpenBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, OversetBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, SymmetryBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, ABLTopBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, PeriodicBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, NonConformalBoundaryConditionData& rhs);

void operator>>(const YAML::Node& node, ConstantInitialConditionData& rhs);

void operator>>(const YAML::Node& node, UserFunctionInitialConditionData& rhs);

void operator>>(const YAML::Node& node, std::map<std::string, bool>& mapName);
void operator>>(const YAML::Node& node, std::map<std::string, double>& mapName);
void
operator>>(const YAML::Node& node, std::map<std::string, std::string>& mapName);
void operator>>(
  const YAML::Node& node,
  std::map<std::string, std::vector<std::string>>& mapName);
void operator>>(
  const YAML::Node& node, std::map<std::string, std::vector<double>>& mapName);

bool case_insensitive_compare(std::string s1, std::string s2);

} // namespace nalu
} // namespace sierra

namespace YAML {

template <>
struct convert<sierra::nalu::Velocity>
{
  static bool decode(const Node& node, sierra::nalu::Velocity& rhs);
};

template <>
struct convert<sierra::nalu::Coordinates>
{
  static bool decode(const Node& node, sierra::nalu::Coordinates& rhs);
};

template <>
struct convert<sierra::nalu::Pressure>
{
  static bool decode(const Node& node, sierra::nalu::Pressure& rhs);
};

template <>
struct convert<sierra::nalu::TurbKinEnergy>
{
  static bool decode(const Node& node, sierra::nalu::TurbKinEnergy& rhs);
};

template <>
struct convert<sierra::nalu::SpecDissRate>
{
  static bool decode(const Node& node, sierra::nalu::SpecDissRate& rhs);
};

template <>
struct convert<sierra::nalu::TotDissRate>
{
  static bool decode(const Node& node, sierra::nalu::TotDissRate& rhs);
};

template <>
struct convert<sierra::nalu::GammaInf>
{
  static bool decode(const Node& node, sierra::nalu::GammaInf& rhs);
};

template <>
struct convert<sierra::nalu::Temperature>
{
  static bool decode(const Node& node, sierra::nalu::Temperature& rhs);
};

template <>
struct convert<sierra::nalu::MixtureFraction>
{
  static bool decode(const Node& node, sierra::nalu::MixtureFraction& rhs);
};

template <>
struct convert<sierra::nalu::MassFraction>
{
  static bool decode(const Node& node, sierra::nalu::MassFraction& rhs);
};

template <>
struct convert<sierra::nalu::ReferenceTemperature>
{
  static bool decode(const Node& node, sierra::nalu::ReferenceTemperature& rhs);
};

template <>
struct convert<sierra::nalu::UserData>
{
  static bool decode(const Node& node, sierra::nalu::UserData& rhs);
};

template <>
struct convert<sierra::nalu::RoughnessHeight>
{
  static bool decode(const Node& node, sierra::nalu::RoughnessHeight& z0);
};

template <>
struct convert<sierra::nalu::NormalHeatFlux>
{
  static bool decode(const Node& node, sierra::nalu::NormalHeatFlux& rhs);
};

template <>
struct convert<sierra::nalu::NormalTemperatureGradient>
{
  static bool
  decode(const Node& node, sierra::nalu::NormalTemperatureGradient& rhs);
};

template <>
struct convert<sierra::nalu::MasterSlave>
{
  static bool decode(const Node& node, sierra::nalu::MasterSlave& rhs);
};

template <>
struct convert<sierra::nalu::WallUserData>
{
  static bool decode(const Node& node, sierra::nalu::WallUserData& rhs);
};

template <>
struct convert<sierra::nalu::InflowUserData>
{
  static bool decode(const Node& node, sierra::nalu::InflowUserData& rhs);
};

template <>
struct convert<sierra::nalu::OpenUserData>
{
  static bool decode(const Node& node, sierra::nalu::OpenUserData& rhs);
};

template <>
struct convert<sierra::nalu::OversetUserData>
{
  static bool decode(const Node& node, sierra::nalu::OversetUserData& rhs);
};

template <>
struct convert<sierra::nalu::SymmetryUserData>
{
  static bool decode(const Node& node, sierra::nalu::SymmetryUserData& rhs);
};

template <>
struct convert<sierra::nalu::ABLTopUserData>
{
  static bool decode(const Node& node, sierra::nalu::ABLTopUserData& rhs);
};

template <>
struct convert<sierra::nalu::PeriodicUserData>
{
  static bool decode(const Node& node, sierra::nalu::PeriodicUserData& rhs);
};

template <>
struct convert<sierra::nalu::NonConformalUserData>
{
  static bool decode(const Node& node, sierra::nalu::NonConformalUserData& rhs);
};

template <>
struct convert<sierra::nalu::BoundaryConditionOptions>
{
  static bool
  decode(const Node& node, sierra::nalu::BoundaryConditionOptions& rhs);
};

template <>
struct convert<sierra::nalu::MeshInput>
{
  static bool decode(const Node& node, sierra::nalu::MeshInput& rhs);
};

template <>
struct convert<std::map<std::string, std::vector<std::string>>>
{
  static bool
  decode(const Node& node, std::map<std::string, std::vector<std::string>>& t);
};

} // namespace YAML

#endif
