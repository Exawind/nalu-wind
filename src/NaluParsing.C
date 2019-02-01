/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

/*------------------------------------------------------------------------*/

#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Simulation.h>
#include <Enums.h>

#include <stk_util/util/ReportHandler.hpp>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <string>
#include <cctype>
#include <algorithm>

namespace sierra
{
  namespace nalu
  {

    const YAML::Node expect_type(const YAML::Node& node, const std::string& key,
      YAML::NodeType::value type, bool optional)
    {
      static std::string types[] =
      { "Undefined", "Null", "Scalar", "Sequence", "Map" };
      std::ostringstream err_msg;

      if (node[key])
      {
        const YAML::Node value = node[key];

        if ((value.Type() != type))
        {
          if (!NaluEnv::self().parallel_rank())
          {
            err_msg << "Error: parsing expected type " << types[type] << " got type = " << types[value.Type()]
            << " for key= " << key
            << " at " << NaluParsingHelper::line_info(node)
            << " node= " << std::endl;
            NaluParsingHelper::emit(err_msg, node);
            err_msg << "Check indentation of input file.";
            std::cout << err_msg.str() << std::endl;
          }
          throw std::runtime_error(err_msg.str());
        }
        return value;
      } else
      {

        if ((!optional))
        {
          if (!NaluEnv::self().parallel_rank())
          {
            err_msg << "Error: parsing expected required value " << key
            << " but it was not found at"
            << NaluParsingHelper::line_info(node) << " for Node= "
            << std::endl;
            NaluParsingHelper::emit(err_msg, node);
            std::cout << err_msg.str() << std::endl;
          }
          throw std::runtime_error(err_msg.str());
        }

        return node[key];
      }

    }

    const YAML::Node expect_null(const YAML::Node& node, const std::string& key,
      bool optional)
    {
      return expect_type(node, key, YAML::NodeType::Null, optional);
    }
    const YAML::Node expect_scalar(const YAML::Node& node,
      const std::string& key, bool optional)
    {
      return expect_type(node, key, YAML::NodeType::Scalar, optional);
    }
    const YAML::Node expect_sequence(const YAML::Node& node,
      const std::string& key, bool optional)
    {
      return expect_type(node, key, YAML::NodeType::Sequence, optional);
    }
    const YAML::Node expect_map(const YAML::Node& node, const std::string& key,
      bool optional)
    {
      return expect_type(node, key, YAML::NodeType::Map, optional);
    }

    void operator >>(const YAML::Node& node, WallBoundaryConditionData& wallBC)
    {
      wallBC.bcName_ = node["wall_boundary_condition"].as<std::string>();
      wallBC.targetName_ = node["target_name"].as<std::string>();
      wallBC.theBcType_ = WALL_BC;
      const YAML::Node wallUserData = node["wall_user_data"];
      wallBC.userData_ = wallUserData.as<WallUserData>();
      // check for typical rogue line command
      if (node["user_function_name"])
        throw std::runtime_error(
          "user_function_data is misplaced; it must be under wall_user_data");
    }

    void operator >>(const YAML::Node& node,
      InflowBoundaryConditionData& inflowBC)
    {
      inflowBC.bcName_ = node["inflow_boundary_condition"].as<std::string>();
      inflowBC.targetName_ = node["target_name"].as<std::string>();
      inflowBC.theBcType_ = INFLOW_BC;
      const YAML::Node& inflowUserData = node["inflow_user_data"];
      inflowBC.userData_ = inflowUserData.as<InflowUserData>();
      // check for typical rogue line command
      if (node["user_function_name"])
        throw std::runtime_error(
          "user_function_data is misplaced; it must be under inflow_user_data");
    }

    void operator >>(const YAML::Node& node, OpenBoundaryConditionData& openBC)
    {
      openBC.bcName_ = node["open_boundary_condition"].as<std::string>();
      openBC.targetName_ = node["target_name"].as<std::string>();
      openBC.theBcType_ = OPEN_BC;
      const YAML::Node& openUserData = node["open_user_data"];
      openBC.userData_ = openUserData.as<OpenUserData>();
      // check for typical rogue line command
      if (node["user_function_name"])
        throw std::runtime_error(
          "user_function_data is misplaced; it must be under open_user_data");
    }

    void operator >>(const YAML::Node& node,
      OversetBoundaryConditionData& oversetBC)
    {
      oversetBC.bcName_ = node["overset_boundary_condition"].as<std::string>();
      oversetBC.theBcType_ = OVERSET_BC;
      oversetBC.oversetConnectivityType_ =
          OversetBoundaryConditionData::NALU_STK;

      // Determine the API to be used; we have already set the default
      if (node["overset_connectivity_type"])
      {
        std::string ogaName =
            node["overset_connectivity_type"].as<std::string>();

        if (ogaName == "nalu_stk")
        {
          oversetBC.oversetConnectivityType_ =
              OversetBoundaryConditionData::NALU_STK;
        } else if (ogaName == "tioga")
        {
#ifdef NALU_USES_TIOGA
          oversetBC.oversetConnectivityType_ = OversetBoundaryConditionData::TPL_TIOGA;
#else
          throw std::runtime_error(
            "TIOGA overset connectivity requested in input file. "
              "However, the optional TPL was not included during compile time.");
#endif
        } else
        {
          throw std::runtime_error(
            "Nalu supports two overset connectivity packages: 'nalu_stk' and 'tioga'. "
                "Value in input file: " + ogaName);
        }
      }

      const YAML::Node& oversetUserData = node["overset_user_data"];

      switch (oversetBC.oversetConnectivityType_)
      {
        case OversetBoundaryConditionData::NALU_STK:
          oversetBC.userData_ = oversetUserData.as<OversetUserData>();
          break;

        case OversetBoundaryConditionData::TPL_TIOGA:
#ifdef NALU_USES_TIOGA
          oversetBC.userData_.oversetBlocks_ = oversetUserData;
#else
          throw std::runtime_error(
            "TIOGA TPL support not enabled during compilation phase.");
#endif
          break;

        case OversetBoundaryConditionData::OVERSET_NONE:
        default:
          throw std::runtime_error(
            "Invalid overset connectivity setting in input file.");
          break;
      }
    }

    void operator >>(const YAML::Node& node,
      SymmetryBoundaryConditionData& symmetryBC)
    {
      symmetryBC.bcName_ =
          node["symmetry_boundary_condition"].as<std::string>();
      symmetryBC.targetName_ = node["target_name"].as<std::string>();
      symmetryBC.theBcType_ = SYMMETRY_BC;
      const YAML::Node& symmetryUserData = node["symmetry_user_data"];
      symmetryBC.userData_ = symmetryUserData.as<SymmetryUserData>();
    }

    void operator >>(const YAML::Node& node,
      ABLTopBoundaryConditionData& abltopBC)
    {
      abltopBC.bcName_ =
          node["abltop_boundary_condition"].as<std::string>();
      abltopBC.targetName_ = node["target_name"].as<std::string>();
      abltopBC.theBcType_ = ABLTOP_BC;
      const YAML::Node& abltopUserData = node["abltop_user_data"];
      abltopBC.userData_ = abltopUserData.as<ABLTopUserData>();
    }

    void operator >>(const YAML::Node& node,
      PeriodicBoundaryConditionData& periodicBC)
    {
      periodicBC.bcName_ =
          node["periodic_boundary_condition"].as<std::string>();
      periodicBC.masterSlave_ = node["target_name"].as<MasterSlave>();
      periodicBC.targetName_ = periodicBC.masterSlave_.master_ + "_"
          + periodicBC.masterSlave_.slave_;
      periodicBC.theBcType_ = PERIODIC_BC;
      const YAML::Node& periodicUserData = node["periodic_user_data"];
      periodicBC.userData_ = periodicUserData.as<PeriodicUserData>();
    }

    void operator >>(const YAML::Node& node,
      NonConformalBoundaryConditionData& nonConformalBC)
    {
      nonConformalBC.bcName_ = node["non_conformal_boundary_condition"].as<
          std::string>();
      nonConformalBC.theBcType_ = NON_CONFORMAL_BC;
      const YAML::Node& nonConformalUserData = node["non_conformal_user_data"];
      nonConformalBC.userData_ =
          nonConformalUserData.as<NonConformalUserData>();

      // check for old/new syntax; one of the two is required to define the current/opposing part(s)
      const YAML::Node targetPairName = node["target_name"];
      const YAML::Node targetsCurrent = node["current_target_name"];
      const YAML::Node targetsOpposing = node["opposing_target_name"];

      if (targetPairName)
      {
        if (targetPairName.size() != 2)
          throw std::runtime_error(
            "For non-conformal algorithms, ne must specify two targets, e.g., [surf_1, surf_2]");
        nonConformalBC.currentPartNameVec_.resize(1);
        nonConformalBC.opposingPartNameVec_.resize(1);
        nonConformalBC.currentPartNameVec_[0] =
            targetPairName[0].as<std::string>();
        nonConformalBC.opposingPartNameVec_[0] = targetPairName[1].as<
            std::string>();
      } else
      {
        if (!targetsCurrent)
          throw std::runtime_error(
            "NonConformal::Error: part definition error: missing current_target_name");
        if (!targetsOpposing)
          throw std::runtime_error(
            "NonConformal::Error: part definition error: missing opposing_target_name");

        // set current
        if (targetsCurrent.Type() == YAML::NodeType::Scalar)
        {
          nonConformalBC.currentPartNameVec_.resize(1);
          nonConformalBC.currentPartNameVec_[0] =
              targetsCurrent.as<std::string>();
        } else
        {
          nonConformalBC.currentPartNameVec_.resize(targetsCurrent.size());
          for (size_t i = 0; i < targetsCurrent.size(); ++i)
          {
            nonConformalBC.currentPartNameVec_[i] = targetsCurrent[i].as<
                std::string>();
          }
        }

        // set opposing
        if (targetsOpposing.Type() == YAML::NodeType::Scalar)
        {
          nonConformalBC.opposingPartNameVec_.resize(1);
          nonConformalBC.opposingPartNameVec_[0] = targetsOpposing.as<
              std::string>();
        } else
        {
          nonConformalBC.opposingPartNameVec_.resize(targetsOpposing.size());
          for (size_t i = 0; i < targetsOpposing.size(); ++i)
          {
            nonConformalBC.opposingPartNameVec_[i] = targetsOpposing[i].as<
                std::string>();
          }
        }
      }

      // set target name for debug
      std::string debugName = "_Current_";
      for (size_t k = 0; k < nonConformalBC.currentPartNameVec_.size(); ++k)
        debugName += nonConformalBC.currentPartNameVec_[k] + "_";
      debugName += "_Opposing_";
      for (size_t k = 0; k < nonConformalBC.opposingPartNameVec_.size(); ++k)
        debugName += nonConformalBC.opposingPartNameVec_[k] + "_";
      nonConformalBC.targetName_ = debugName;
    }

    void operator >>(const YAML::Node& node,
      UserFunctionInitialConditionData& fcnIC)
    {

      fcnIC.theIcType_ = sierra::nalu::FUNCTION_UD;
      fcnIC.icName_ = node["user_function"].as<std::string>();
      const YAML::Node & targets = node["target_name"];
      if (targets.Type() == YAML::NodeType::Scalar)
      {
        fcnIC.targetNames_.resize(1);
        fcnIC.targetNames_[0] = targets.as<std::string>();
      } else
      {
        fcnIC.targetNames_.resize(targets.size());
        for (size_t i = 0; i < targets.size(); ++i)
        {
          fcnIC.targetNames_[i] = targets[i].as<std::string>();
        }
      }

      // extract function name and parameters
      if (expect_map(node, "user_function_name", false))
      {
        fcnIC.functionNames_ = node["user_function_name"].as<
            std::map<std::string, std::string> >();
      }

      if (expect_map(node, "user_function_parameters", true))
      {
        fcnIC.functionParams_ = node["user_function_parameters"].as<
            std::map<std::string, std::vector<double> > >();
      }

    }

    void operator >>(const YAML::Node& node,
      ConstantInitialConditionData& constIC)
    {

      constIC.theIcType_ = sierra::nalu::CONSTANT_UD;
      constIC.icName_ = node["constant"].as<std::string>();
      const YAML::Node & targets = node["target_name"];

      if (targets.Type() == YAML::NodeType::Scalar)
      {
        constIC.targetNames_.resize(1);
        constIC.targetNames_[0] = targets.as<std::string>();
        NaluEnv::self().naluOutputP0() << "constant IC: name: " << constIC.icName_ << " , target[" << 0 << "] = " << constIC.targetNames_[0] << std::endl;
        if (constIC.targetNames_[0].find(',') != std::string::npos)
        {
          throw std::runtime_error(
            "In " + constIC.icName_
                + " found ',' in target name - you must enclose in '[...]' for multiple targets");
        }

      } else
      {
        constIC.targetNames_.resize(targets.size());
        for (size_t i = 0; i < targets.size(); ++i)
        {
          constIC.targetNames_[i] = targets[i].as<std::string>();
          if (constIC.root()->debug())
            NaluEnv::self().naluOutputP0() << "constant IC: name: " << constIC.icName_ << " , target[" << i << "] = " << constIC.targetNames_[i] << std::endl;
          }
        }
      const YAML::Node value_node = node["value"];
      size_t value_size = value_node.size();
      constIC.fieldNames_.resize(value_size);
      constIC.data_.resize(value_size);
      if (constIC.root()->debug())
      {
        NaluEnv::self().naluOutputP0() << "fieldNames_.size()= " << constIC.fieldNames_.size()
        << " value.size= " << constIC.data_.size() << std::endl;
      }
      size_t jv = 0;
      for (YAML::const_iterator i = value_node.begin(); i != value_node.end();
          ++i, ++jv)
      {
        const YAML::Node key = i->first;
        const YAML::Node value = i->second;
        constIC.fieldNames_[jv] = key.as<std::string>();
        size_t nvals = value.size();
        if (nvals)
        {
          constIC.data_[jv].resize(nvals);
          for (size_t iv = 0; iv < nvals; ++iv)
          {
            constIC.data_[jv][iv] = value[iv].as<double>();
            if (constIC.root()->debug())
            {
              NaluEnv::self().naluOutputP0() << "fieldNames_= " << constIC.fieldNames_[jv] << " value= "
              << constIC.data_[jv][iv] << std::endl;
            }
          }
        } else
        {
          constIC.data_[jv].resize(1);
          constIC.data_[jv][0] = value.as<double>();
          if (constIC.root()->debug())
          {
            NaluEnv::self().naluOutputP0() << "fieldNames_= " << constIC.fieldNames_[jv] << " value= " << constIC.data_[jv][0] << std::endl;
          }
        }

      }
    }

    void operator >>(const YAML::Node& node,
      std::map<std::string, bool> & mapName)
    {
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i)
      {
        const YAML::Node & key = i->first;
        const YAML::Node & value = i->second;
        std::string stringName;
        stringName = key.as<std::string>();
        bool data;
        data = value.as<bool>();
        mapName[stringName] = data;
      }
    }

    void operator >>(const YAML::Node& node,
      std::map<std::string, double> & mapName)
    {
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i)
      {
        const YAML::Node & key = i->first;
        const YAML::Node & value = i->second;
        std::string stringName;
        stringName = key.as<std::string>();
        double data;
        data = value.as<double>();
        mapName[stringName] = data;
      }
    }

    void operator >>(const YAML::Node& node,
      std::map<std::string, std::string> & mapName)
    {
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i)
      {
        const YAML::Node & key = i->first;
        const YAML::Node & value = i->second;
        std::string stringName;
        stringName = key.as<std::string>();
        std::string data;
        data = value.as<std::string>();
        mapName[stringName] = data;
      }
    }

    void operator >>(const YAML::Node& node,
      std::map<std::string, std::vector<std::string> >& mapName)
    {
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i)
      {
        const YAML::Node & key = i->first;
        const YAML::Node & targets = i->second;
        std::string stringName;
        stringName = key.as<std::string>();

        std::vector < std::string > &vecOfStrings = mapName[stringName];
        std::string theName;
        if (targets.Type() == YAML::NodeType::Scalar)
        {
          theName = targets.as<std::string>();
          vecOfStrings.push_back(theName);
        } else
        {
          for (size_t it = 0; it < targets.size(); ++it)
          {
            theName = targets[it].as<std::string>();
            vecOfStrings.push_back(theName);
          }
        }
      }
    }

    void operator >>(const YAML::Node& node,
      std::map<std::string, std::vector<double> >& mapName)
    {
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i)
      {
        const YAML::Node & key = i->first;
        const YAML::Node & targets = i->second;
        std::string stringName;
        stringName = key.as<std::string>();

        std::vector<double> &vecOfDoubles = mapName[stringName];
        double value;
        if (targets.Type() == YAML::NodeType::Scalar)
        {
          value = targets.as<double>();
          vecOfDoubles.push_back(value);
        } else
        {
          for (size_t it = 0; it < targets.size(); ++it)
          {
            value = targets[it].as<double>();
            vecOfDoubles.push_back(value);
          }
        }
      }
    }

    bool case_insensitive_compare(std::string s1, std::string s2)
    {
      std::transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
      std::transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
      return (s1 == s2);
    }

  } // namespace nalu
} // namespace Sierra

namespace YAML
{

  bool convert<sierra::nalu::Velocity>::decode(const Node& node,
    sierra::nalu::Velocity& v)
  {
    if (!node.IsSequence() || node.size() < 2)
    {
      return false;
    }

    v.ux_ = node[0].as<double>();
    v.uy_ = node[1].as<double>();
    if (node.size() > 2)
      v.uz_ = node[2].as<double>();

    return true;
  }

  bool convert<sierra::nalu::Coordinates>::decode(const Node& node,
    sierra::nalu::Coordinates& cx)
  {
    if (!node.IsSequence() || node.size() < 2)
    {
      return false;
    }

    cx.x_ = node[0].as<double>();
    cx.y_ = node[1].as<double>();
    if (node.size() > 2)
      cx.z_ = node[2].as<double>();

    return true;
  }

  bool convert<sierra::nalu::Pressure>::decode(const Node& node,
    sierra::nalu::Pressure& p)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    p.pressure_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::TurbKinEnergy>::decode(const Node& node,
    sierra::nalu::TurbKinEnergy& tke)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    tke.turbKinEnergy_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::SpecDissRate>::decode(const Node& node,
    sierra::nalu::SpecDissRate& sdr)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    sdr.specDissRate_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::Temperature>::decode(const Node& node,
    sierra::nalu::Temperature& t)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    t.temperature_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::MixtureFraction>::decode(const Node& node,
    sierra::nalu::MixtureFraction& z)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    z.mixFrac_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::MassFraction>::decode(const Node& node,
    sierra::nalu::MassFraction& yk)
  {
    if (!node.IsSequence())
    {
      return false;
    }

    yk.massFraction_.resize(node.size());
    size_t ykSize = node.size();
    for (size_t k = 0; k < ykSize; ++k)
    {
      yk.massFraction_[k] = node[k].as<double>();
    }

    return true;
  }

  bool convert<sierra::nalu::Emissivity>::decode(const Node& node,
    sierra::nalu::Emissivity& emiss)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    emiss.emissivity_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::Irradiation>::decode(const Node& node,
    sierra::nalu::Irradiation& irrad)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    irrad.irradiation_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::Transmissivity>::decode(const Node& node,
    sierra::nalu::Transmissivity& tmiss)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    tmiss.transmissivity_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::EnvironmentalT>::decode(const Node& node,
    sierra::nalu::EnvironmentalT& et)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    et.environmentalT_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::RoughnessHeight>::decode(const Node& node,
    sierra::nalu::RoughnessHeight& z0)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    z0.z0_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::NormalHeatFlux>::decode(const Node& node,
    sierra::nalu::NormalHeatFlux& q)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    q.qn_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::NormalTemperatureGradient>::decode(const Node& node,
    sierra::nalu::NormalTemperatureGradient& tempGrad)
  {
    if (!node.IsScalar())
    {
      return false;
    }
   
    tempGrad.tempGradN_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::ReferenceTemperature>::decode(const Node& node,
    sierra::nalu::ReferenceTemperature& rt)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    rt.referenceTemperature_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::HeatTransferCoefficient>::decode(const Node& node,
    sierra::nalu::HeatTransferCoefficient& htc)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    htc.heatTransferCoefficient_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::RobinCouplingParameter>::decode(const Node& node,
    sierra::nalu::RobinCouplingParameter& alpha)
  {
    if (!node.IsScalar())
    {
      return false;
    }

    alpha.robinCouplingParameter_ = node.as<double>();

    return true;
  }

  bool convert<sierra::nalu::WallUserData>::decode(const Node& node,
    sierra::nalu::WallUserData& wallData)
  {

    // constant data; all optional
    if (node["velocity"])
    {
      wallData.u_ = node["velocity"].as<sierra::nalu::Velocity>();
      wallData.bcDataSpecifiedMap_["velocity"] = true;
      wallData.bcDataTypeMap_["velocity"] = sierra::nalu::CONSTANT_UD;
    }

    if (node["mesh_displacement"])
    {
      wallData.dx_ = node["mesh_displacement"].as<sierra::nalu::Velocity>();
      wallData.bcDataSpecifiedMap_["mesh_displacement"] = true;
      wallData.bcDataTypeMap_["mesh_displacement"] = sierra::nalu::CONSTANT_UD;
    }
    if (node["turbulent_ke"])
    {
      wallData.tke_ = node["turbulent_ke"].as<sierra::nalu::TurbKinEnergy>();
      wallData.bcDataSpecifiedMap_["turbulent_ke"] = true;
      wallData.bcDataTypeMap_["turbulent_ke"] = sierra::nalu::CONSTANT_UD;
    }
    if (node["temperature"])
    {
      wallData.temperature_ =
          node["temperature"].as<sierra::nalu::Temperature>();
      wallData.bcDataSpecifiedMap_["temperature"] = true;
      wallData.bcDataTypeMap_["temperature"] = sierra::nalu::CONSTANT_UD;
      wallData.tempSpec_ = true;
    }

    if (node["mixture_fraction"])
    {
      wallData.mixFrac_ = node["mixture_fraction"].as<
          sierra::nalu::MixtureFraction>();
      wallData.bcDataSpecifiedMap_["mixture_fraction"] = true;
      wallData.bcDataTypeMap_["mixture_fraction"] = sierra::nalu::CONSTANT_UD;
    }

    if (node["mass_fraction"])
    {
      wallData.massFraction_ = node["mass_fraction"].as<
          sierra::nalu::MassFraction>();
      wallData.bcDataSpecifiedMap_["mass_fraction"] = true;
      wallData.bcDataTypeMap_["mass_fraction"] = sierra::nalu::CONSTANT_UD;
    }
    if (node["emissivity"])
    {
      wallData.emissivity_ = node["emissivity"].as<sierra::nalu::Emissivity>();
      wallData.emissSpec_ = true;
    }
    if (node["transmissivity"])
    {
      wallData.transmissivity_ = node["transmissivity"].as<
          sierra::nalu::Transmissivity>();
    }
    if (node["environmental_temperature"])
    {
      wallData.environmentalT_ = node["environmental_temperature"].as<
          sierra::nalu::EnvironmentalT>();
    }
    if (node["adiabatic"])
    {
      wallData.isAdiabatic_ = node["adiabatic"].as<bool>();
    }
    if (node["interface"])
    {
      wallData.isInterface_ = node["interface"].as<bool>();
    }

    if (node["reference_temperature"])
    {
      wallData.referenceTemperature_ = node["reference_temperature"].as<
          sierra::nalu::ReferenceTemperature>();
      wallData.refTempSpec_ = true;
    }
    if (node["gravity_vector_component"])
    {
      wallData.gravityComponent_ =
          node["gravity_vector_component"].as<unsigned>();
    }
    if (node["roughness_height"])
    {
      wallData.z0_ =
          node["roughness_height"].as<sierra::nalu::RoughnessHeight>();
    }
    if (node["heat_transfer_coefficient"])
    {
      wallData.heatTransferCoefficient_ = node["heat_transfer_coefficient"].as<
          sierra::nalu::HeatTransferCoefficient>();
      wallData.htcSpec_ = true;
    }
    if (node["irradiation"])
    {
      wallData.irradiation_ =
          node["irradiation"].as<sierra::nalu::Irradiation>();
      wallData.irradSpec_ = true;
    }
    if (node["robin_coupling_parameter"])
    {
      wallData.robinCouplingParameter_ = node["robin_coupling_parameter"].as<
          sierra::nalu::RobinCouplingParameter>();
      wallData.robinParameterSpec_ = true;
    }
    if (node["use_wall_function"])
    {
      wallData.wallFunctionApproach_ = node["use_wall_function"].as<bool>();
    }
    if (node["use_abl_wall_function"])
    {
      wallData.wallFunctionApproach_ = node["use_abl_wall_function"].as<bool>();
      wallData.ablWallFunctionApproach_ =
          node["use_abl_wall_function"].as<bool>();
    }
    if (node["pressure"])
    {
      wallData.pressure_ = node["pressure"].as<sierra::nalu::Pressure>();
      wallData.bcDataSpecifiedMap_["pressure"] = true;
      wallData.bcDataTypeMap_["pressure"] = sierra::nalu::CONSTANT_UD;
    }
    if (node["fsi_interface"])
    {
      wallData.isFsiInterface_ = node["fsi_interface"].as<bool>();
    }

    // not appropriate
    if (node["specific_dissipation_rate"])
    {
      throw std::runtime_error(
        "specific_dissipation rate at walls is provided by a model, not the user");
    }

    if (node["heat_flux"])
    {
      wallData.q_ = node["heat_flux"].as<sierra::nalu::NormalHeatFlux>();
      wallData.heatFluxSpec_ = true;
    }

    // function data
    const bool optional = true;
    const Node userFcnNode = sierra::nalu::expect_map(node,
      "user_function_name", optional);
    if (userFcnNode)
    {
      for (const_iterator i = userFcnNode.begin(); i != userFcnNode.end(); ++i)
      {
        const Node & key = i->first;
        const Node & value = i->second;
        std::string stringName = key.as<std::string>();
        std::string data = value.as<std::string>();
        wallData.bcDataSpecifiedMap_[stringName] = true;
        wallData.bcDataTypeMap_[stringName] = sierra::nalu::FUNCTION_UD;
        wallData.userFunctionMap_[stringName] = data;
      }

      // extract function name and parameters
      if (sierra::nalu::expect_map(node, "user_function_parameters", true))
      {
        wallData.functionParams_ = node["user_function_parameters"].as<
            std::map<std::string, std::vector<double> > >();
      }

      // extract function name and string parameters
      if (sierra::nalu::expect_map(node, "user_function_string_parameters",
        true))
      {
        wallData.functionStringParams_ =
            node["user_function_string_parameters"].as<
                std::map<std::string, std::vector<std::string> > >();
      }
    }

    return true;

  }

  bool convert<sierra::nalu::MasterSlave>::decode(const Node& node,
    sierra::nalu::MasterSlave& ms)
  {

    if (!node.IsSequence() || node.size() != 2)
    {
      return false;
    }

    ms.master_ = node[0].as<std::string>();
    ms.slave_ = node[1].as<std::string>();

    return true;
  }

  bool convert<sierra::nalu::InflowUserData>::decode(const Node& node,
    sierra::nalu::InflowUserData& inflowData)
  {
    // optional
    if (node["velocity"])
    {
      inflowData.u_ = node["velocity"].as<sierra::nalu::Velocity>();
      inflowData.uSpec_ = true;
    }
    if (node["turbulent_ke"])
    {
      inflowData.tke_ = node["turbulent_ke"].as<sierra::nalu::TurbKinEnergy>();
      inflowData.tkeSpec_ = true;
    }
    if (node["specific_dissipation_rate"])
    {
      inflowData.sdr_ = node["specific_dissipation_rate"].as<
          sierra::nalu::SpecDissRate>();
      inflowData.sdrSpec_ = true;
    }
    if (node["mixture_fraction"])
    {
      inflowData.mixFrac_ = node["mixture_fraction"].as<
          sierra::nalu::MixtureFraction>();
      inflowData.mixFracSpec_ = true;
    }
    if (node["mass_fraction"])
    {
      inflowData.massFraction_ = node["mass_fraction"].as<
          sierra::nalu::MassFraction>();
      inflowData.massFractionSpec_ = true;
    }
    if (node["temperature"])
    {
      inflowData.temperature_ =
          node["temperature"].as<sierra::nalu::Temperature>();
      inflowData.tempSpec_ = true;
    }

    const bool optional = true;
    const Node userFcnNode = sierra::nalu::expect_map(node,
      "user_function_name", optional);
    if (userFcnNode)
    {
      for (const_iterator i = userFcnNode.begin(); i != userFcnNode.end(); ++i)
      {
        const Node & key = i->first;
        const Node & value = i->second;
        std::string stringName;
        stringName = key.as<std::string>();
        std::string data;
        data = value.as<std::string>();
        inflowData.bcDataSpecifiedMap_[stringName] = true;
        inflowData.bcDataTypeMap_[stringName] = sierra::nalu::FUNCTION_UD;
        inflowData.userFunctionMap_[stringName] = data;
      }

      // extract function name and parameters
      if (sierra::nalu::expect_map(node, "user_function_parameters", true))
      {
        inflowData.functionParams_ = node["user_function_parameters"].as<
            std::map<std::string, std::vector<double> > >();
      }
    }

    // check for external data
    if (node["external_data"])
    {
      inflowData.externalData_ = node["external_data"].as<bool>();
    }

    return true;
  }

  bool convert<sierra::nalu::OpenUserData>::decode(const Node& node,
    sierra::nalu::OpenUserData& openData)
  {

    // optional
    if (node["velocity"])
    {
      openData.u_ = node["velocity"].as<sierra::nalu::Velocity>();
      openData.uSpec_ = true;
    }
    // optional
    if (node["turbulent_ke"])
    {
      openData.tke_ = node["turbulent_ke"].as<sierra::nalu::TurbKinEnergy>();
      openData.tkeSpec_ = true;
    }
    if (node["specific_dissipation_rate"])
    {
      openData.sdr_ = node["specific_dissipation_rate"].as<
          sierra::nalu::SpecDissRate>();
      openData.sdrSpec_ = true;
    }
    if (node["pressure"])
    {
      openData.p_ = node["pressure"].as<sierra::nalu::Pressure>();
      openData.pSpec_ = true;
    }
    if (node["mixture_fraction"])
    {
      openData.mixFrac_ = node["mixture_fraction"].as<
          sierra::nalu::MixtureFraction>();
      openData.mixFracSpec_ = true;
    }
    if (node["mass_fraction"])
    {
      openData.massFraction_ = node["mass_fraction"].as<
          sierra::nalu::MassFraction>();
      openData.massFractionSpec_ = true;
    }
    if (node["temperature"])
    {
      openData.temperature_ =
          node["temperature"].as<sierra::nalu::Temperature>();
      openData.tempSpec_ = true;
    }

    return true;
  }

  bool convert<sierra::nalu::OversetUserData>::decode(const Node& node,
    sierra::nalu::OversetUserData& oversetData)
  {
    // nothing is optional
    if (node["percent_overlap"])
    {
      oversetData.percentOverlap_ = node["percent_overlap"].as<double>();
    } else
    {
      throw std::runtime_error("One MUST specify overset overlap percentage");
    }

    if (node["background_block"])
    {
      oversetData.backgroundBlock_ = node["background_block"].as<std::string>();
    } else
    {
      throw std::runtime_error("One MUST specify background block");
    }

    if (node["overset_block"])
    {
      const Node oversetBlock = node["overset_block"];
      if (oversetBlock.Type() == NodeType::Scalar)
      {
        oversetData.oversetBlockVec_.resize(1);
        oversetData.oversetBlockVec_[0] = oversetBlock.as<std::string>();
      } else
      {
        oversetData.oversetBlockVec_.resize(oversetBlock.size());
        oversetData.oversetBlockVec_ =
            oversetBlock.as<std::vector<std::string>>();
      }
    } else
    {
      throw std::runtime_error("One MUST specify overset block(s)");
    }

    if (node["background_cut_block"])
    {
      oversetData.backgroundCutBlock_ = node["background_cut_block"].as<
          std::string>();
    } else
    {
      throw std::runtime_error("One MUST specify background cut block");
    }

    if (node["background_cut_surface"])
    {
      oversetData.backgroundSurface_ = node["background_cut_surface"].as<
          std::string>();
    } else
    {
      throw std::runtime_error("One MUST specify background cut surface");
    }

    if (node["overset_surface"])
    {
      oversetData.oversetSurface_ = node["overset_surface"].as<std::string>();
    } else
    {
      throw std::runtime_error("One MUST specify overset surface");
    }

    if (node["clip_isoparametric_coordinates"])
    {
      oversetData.clipIsoParametricCoords_ =
          node["clip_isoparametric_coordinates"].as<bool>();
    }

    if (node["detailed_output"])
    {
      oversetData.detailedOutput_ = node["detailed_output"].as<bool>();
    }

    return true;
  }

  bool convert<sierra::nalu::SymmetryUserData>::decode(const Node& node,
    sierra::nalu::SymmetryUserData&  /* symmetryData */)
  {
    // Normal temperature gradient for ABL top BC is now implemented in
    // abltop_boundary_condition. Throw error to inform user of the change
    if (node["normal_temperature_gradient"])
      throw std::runtime_error("SymmetryBoundaryConditionData: Normal temperature gradient for ABL top boundary must use abltop_boundary_condition");
    return true;
  }

  bool convert<sierra::nalu::ABLTopUserData>::decode(const Node& node,
    sierra::nalu::ABLTopUserData& abltopData)
  {
    // This allows the user to set a fixed noraml temperature gradient that is
    // achieved through application of a compatible normal  heat flux. 
    if (node["normal_temperature_gradient"])
    {
      abltopData.normalTemperatureGradient_ = node["normal_temperature_gradient"].as<sierra::nalu::NormalTemperatureGradient>();
      abltopData.normalTemperatureGradientSpec_ = true;
    }
    abltopData.ABLTopBC_ = false;
    if ( node["potential_flow_bc"] ) {
      abltopData.ABLTopBC_ = node["potential_flow_bc"].as<bool>();
    }
    if (abltopData.ABLTopBC_) {
      abltopData.grid_dims_ = node["grid_dimensions"].as<std::vector<int>>();
      abltopData.horiz_bcs_ = node["horizontal_bcs"].as<std::vector<int>>();
      if ( node["z_sample"] ) {
        abltopData.z_sample_  = node["z_sample"].as<double>();
      }
    }
    return true;
  }

  bool convert<sierra::nalu::PeriodicUserData>::decode(const Node& node,
    sierra::nalu::PeriodicUserData& periodicData)
  {
    // nothing is optional
    if (node["search_tolerance"])
    {
      periodicData.searchTolerance_ = node["search_tolerance"].as<double>();
    } else
    {
      throw std::runtime_error(
        "One MUST specify search tolerance at periodic bcs");
    }
    if (node["search_method"])
    {
      periodicData.searchMethodName_ = node["search_method"].as<std::string>();
    }

    return true;
  }

  bool convert<sierra::nalu::NonConformalUserData>::decode(const Node& node,
    sierra::nalu::NonConformalUserData& nonConformalData)
  {

    // everything is optional
    if (node["search_method"])
    {
      nonConformalData.searchMethodName_ =
        node["search_method"].as<std::string>();
    }
    if (node["expand_box_percentage"])
    {
      nonConformalData.expandBoxPercentage_ = node["expand_box_percentage"].as<
          double>();
    }
    if (node["clip_isoparametric_coordinates"])
    {
      nonConformalData.clipIsoParametricCoords_ =
          node["clip_isoparametric_coordinates"].as<bool>();
    }
    if (node["search_tolerance"])
    {
      nonConformalData.searchTolerance_ = node["search_tolerance"].as<double>();
    }
    if (node["activate_dynamic_search_algorithm"])
    {
      nonConformalData.dynamicSearchTolAlg_ =
        node["activate_dynamic_search_algorithm"].as<bool>();
    }

    return true;
  }

  bool convert<sierra::nalu::BoundaryConditionOptions>::decode(const Node& node,
    sierra::nalu::BoundaryConditionOptions& bcOptions)
  {

    bcOptions.bcSetName_ = node["boundary_conditions"].as<std::string>();
    node["wall_boundary_condition"] >> bcOptions.wallbc_;
    node["inflow_boundary_condition"] >> bcOptions.inflowbc_;
    node["open_boundary_condition"] >> bcOptions.openbc_;
    node["overset_boundary_condition"] >> bcOptions.oversetbc_;
    node["symmetry_boundary_condition"] >> bcOptions.symmetrybc_;
    node["abltop_boundary_condition"] >> bcOptions.abltopbc_;
    node["periodic_boundary_condition"] >> bcOptions.periodicbc_;
    node["non_confomal_boundary_condition"] >> bcOptions.nonConformalbc_;

    return true;
  }

  bool convert<sierra::nalu::MeshInput>::decode(const Node& node,
    sierra::nalu::MeshInput& meshInput)
  {
    meshInput.meshName_ = node["mesh_name"].as<std::string>();
    return true;
  }

  bool convert<std::map<std::string, std::vector<std::string> > >::decode(
    const YAML::Node& node,
    std::map<std::string, std::vector<std::string> >& mapName)
  {
    for (const_iterator i = node.begin(); i != node.end(); ++i)
    {
      const YAML::Node & key = i->first;
      const YAML::Node & targets = i->second;
      std::string stringName;
      stringName = key.as<std::string>();

      std::vector < std::string > &vecOfStrings = mapName[stringName];
      std::string theName;
      if (targets.Type() == YAML::NodeType::Scalar)
      {
        theName = targets.as<std::string>();
        vecOfStrings.push_back(theName);
      } else
      {
        for (size_t it = 0; it < targets.size(); ++it)
        {
          theName = targets[it].as<std::string>();
          vecOfStrings.push_back(theName);
        }
      }
    }

    return true;
  }

}

