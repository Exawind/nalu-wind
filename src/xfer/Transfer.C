// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <NaluEnv.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>
#include <NaluParsingHelper.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <xfer/Transfer.h>
#include <xfer/Transfers.h>
#include <xfer/FromMesh.h>
#include <xfer/ToMesh.h>
#include <xfer/LinInterp.h>
#include <stk_transfer/GeometricTransfer.hpp>

// stk_search
#include <stk_search/SearchMethod.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// Transfer - base class for Transfer
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Transfer::Transfer(Transfers& transfers)
  : transfers_(transfers),
    couplingPhysicsSpecified_(false),
    transferVariablesSpecified_(false),
    couplingPhysicsName_("none"),
    fromRealm_(NULL),
    toRealm_(NULL),
    name_("none"),
    transferType_("none"),
    transferObjective_("multi_physics"),
    searchMethodName_("stk_kdtree"),
    searchTolerance_(1.0e-4),
    searchExpansionFactor_(1.5)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Transfer::~Transfer()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::load(const YAML::Node& node)
{

  name_ = node["name"].as<std::string>();
  transferType_ = node["type"].as<std::string>();
  if (node["objective"]) {
    transferObjective_ = node["objective"].as<std::string>();
  }

  if (node["coupling_physics"]) {
    couplingPhysicsName_ = node["coupling_physics"].as<std::string>();
    couplingPhysicsSpecified_ = true;
  }

  // realm names
  const YAML::Node realmPair = node["realm_pair"];
  if (realmPair.size() != 2)
    throw std::runtime_error("XFER::Error: need two realm pairs for xfer");
  realmPairName_.first = realmPair[0].as<std::string>();
  realmPairName_.second = realmPair[1].as<std::string>();

  // set bools for variety of mesh part declarations
  const bool hasOld = node["mesh_part_pair"] ? true : false;
  const bool hasNewFrom = node["from_target_name"] ? true : false;
  const bool hasNewTo = node["to_target_name"] ? true : false;

  // mesh part pairs
  if (hasOld) {
    // error check to ensure old and new are not mixed
    if (hasNewFrom || hasNewTo)
      throw std::runtime_error(
        "XFER::Error: part definition error: can not "
        "mix mesh part line commands");

    // proceed safely
    const YAML::Node meshPartPairName = node["mesh_part_pair"];
    if (meshPartPairName.size() != 2)
      throw std::runtime_error("need two mesh part pairs for xfer");
    // resize and set the value
    fromPartNameVec_.resize(1);
    toPartNameVec_.resize(1);
    fromPartNameVec_[0] = meshPartPairName[0].as<std::string>();
    toPartNameVec_[0] = meshPartPairName[1].as<std::string>();
  } else {
    // new methodology that allows for full target; error check
    if (!hasNewFrom)
      throw std::runtime_error(
        "XFER::Error: part definition error: missing a from_target_name");
    if (!hasNewTo)
      throw std::runtime_error(
        "XFER::Error: part definition error: missing a to_target_name");

    // proceed safely; manage "from" parts
    const YAML::Node targetsFrom = node["from_target_name"];
    if (targetsFrom.Type() == YAML::NodeType::Scalar) {
      fromPartNameVec_.resize(1);
      fromPartNameVec_[0] = targetsFrom.as<std::string>();
    } else {
      fromPartNameVec_.resize(targetsFrom.size());
      for (size_t i = 0; i < targetsFrom.size(); ++i) {
        fromPartNameVec_[i] = targetsFrom[i].as<std::string>();
      }
    }

    // manage "to" parts
    const YAML::Node& targetsTo = node["to_target_name"];
    if (targetsTo.Type() == YAML::NodeType::Scalar) {
      toPartNameVec_.resize(1);
      toPartNameVec_[0] = targetsTo.as<std::string>();
    } else {
      toPartNameVec_.resize(targetsTo.size());
      for (size_t i = 0; i < targetsTo.size(); ++i) {
        toPartNameVec_[i] = targetsTo[i].as<std::string>();
      }
    }
  }

  // search method
  if (node["search_method"]) {
    searchMethodName_ = node["search_method"].as<std::string>();
  }

  // search tolerance which forms the initail size of the radius
  if (node["search_tolerance"]) {
    searchTolerance_ = node["search_tolerance"].as<double>();
  }

  // search expansion factor if points are not found
  if (node["search_expansion_factor"]) {
    searchExpansionFactor_ = node["search_expansion_factor"].as<double>();
  }

  // now possible field names
  const YAML::Node y_vars = node["transfer_variables"];
  if (y_vars) {
    transferVariablesSpecified_ = true;
    std::string fromName, toName;
    for (size_t ioption = 0; ioption < y_vars.size(); ioption++) {
      const YAML::Node y_var = y_vars[ioption];
      size_t varPairSize = y_var.size();
      if (varPairSize != 2)
        throw std::runtime_error("need two field name pairs for xfer");
      fromName = y_var[0].as<std::string>();
      toName = y_var[1].as<std::string>();
      transferVariablesPairName_.push_back(std::make_pair(fromName, toName));
    }

    // warn the user...
    NaluEnv::self().naluOutputP0()
      << "Specifying the transfer variables requires expert understanding; "
         "consider using coupling_physics"
      << std::endl;
  }

  // sanity check
  if (couplingPhysicsSpecified_ && transferVariablesSpecified_)
    throw std::runtime_error(
      "physics set and transfer variables specified; "
      "will go with variables specified");

  if (!couplingPhysicsSpecified_ && !transferVariablesSpecified_)
    throw std::runtime_error(
      "neither physics set nor transfer variables specified");

  // now proceed with possible pre-defined transfers
  if (!transferVariablesSpecified_) {
    // hard code for a single type of physics transfer
    if (couplingPhysicsName_ == "fluids_cht") {
      // h
      std::pair<std::string, std::string> thePairH;
      std::string sameNameH = "heat_transfer_coefficient";
      thePairH = std::make_pair(sameNameH, sameNameH);
      transferVariablesPairName_.push_back(thePairH);
      // Too
      std::pair<std::string, std::string> thePairT;
      std::string sameNameT = "reference_temperature";
      thePairT = std::make_pair(sameNameT, sameNameT);
      transferVariablesPairName_.push_back(thePairT);
    } else if (couplingPhysicsName_ == "fluids_robin") {
      // q
      std::pair<std::string, std::string> thePairQ;
      std::string sameNameQ = "normal_heat_flux";
      thePairQ = std::make_pair(sameNameQ, sameNameQ);
      transferVariablesPairName_.push_back(thePairQ);
      // Too
      std::pair<std::string, std::string> thePairT;
      std::string fluidsT = "temperature";
      std::string thermalT = "reference_temperature";
      thePairT = std::make_pair(fluidsT, thermalT);
      transferVariablesPairName_.push_back(thePairT);
      // alpha
      std::pair<std::string, std::string> thePairA;
      std::string sameNameA = "robin_coupling_parameter";
      thePairA = std::make_pair(sameNameA, sameNameA);
      transferVariablesPairName_.push_back(thePairA);
    } else if (couplingPhysicsName_ == "thermal_cht") {
      // T -> T
      std::pair<std::string, std::string> thePairT;
      std::string temperatureName = "temperature";
      thePairT = std::make_pair(temperatureName, temperatureName);
      transferVariablesPairName_.push_back(thePairT);
      // T -> Tbc
      std::pair<std::string, std::string> thePairTbc;
      std::string temperatureBcName = "temperature_bc";
      thePairTbc = std::make_pair(temperatureName, temperatureBcName);
      transferVariablesPairName_.push_back(thePairTbc);
    } else if (couplingPhysicsName_ == "thermal_robin") {
      // T -> T
      std::pair<std::string, std::string> thePairT;
      std::string temperatureName = "temperature";
      thePairT = std::make_pair(temperatureName, temperatureName);
      transferVariablesPairName_.push_back(thePairT);
      // T -> Tbc
      std::pair<std::string, std::string> thePairTbc;
      std::string temperatureBcName = "temperature_bc";
      thePairTbc = std::make_pair(temperatureName, temperatureBcName);
      transferVariablesPairName_.push_back(thePairTbc);
    } else {
      throw std::runtime_error(
        "only supports pre-defined fluids/thermal_cht/robin; perhaps you can "
        "use the generic interface");
    }
  }
}

//--------------------------------------------------------------------------
//-------- breadboard ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::breadboard()
{
  // realm pair
  const std::string fromRealmName = realmPairName_.first;
  const std::string toRealmName = realmPairName_.second;

  // extact the realms
  fromRealm_ = root()->realms_->find_realm(fromRealmName);
  if (NULL == fromRealm_)
    throw std::runtime_error("from realm in xfer is NULL");
  toRealm_ = root()->realms_->find_realm(toRealmName);
  if (NULL == toRealm_)
    throw std::runtime_error("to realm in xfer is NULL");

  // advertise this transfer to realm; for calling control
  fromRealm_->augment_transfer_vector(this, transferObjective_, toRealm_);

  // meta data; bulk data to early to extract?
  stk::mesh::MetaData& fromMetaData = fromRealm_->meta_data();
  stk::mesh::MetaData& toMetaData = toRealm_->meta_data();

  // from mesh parts..
  for (size_t k = 0; k < fromPartNameVec_.size(); ++k) {
    // get the part; no need to subset
    stk::mesh::Part* fromTargetPart =
      fromMetaData.get_part(fromPartNameVec_[k]);
    if (NULL == fromTargetPart)
      throw std::runtime_error(
        "from target part in xfer is NULL; check: " + fromPartNameVec_[k]);
    else
      fromPartVec_.push_back(fromTargetPart);
  }

  // to mesh parts
  for (size_t k = 0; k < toPartNameVec_.size(); ++k) {
    // get the part; no need to subset
    stk::mesh::Part* toTargetPart = toMetaData.get_part(toPartNameVec_[k]);
    if (NULL == toTargetPart)
      throw std::runtime_error(
        "to target part in xfer is NULL; check: " + toPartNameVec_[k]);
    else
      toPartVec_.push_back(toTargetPart);
  }

  // could extract the fields from the realm now and save them off?...
  // FIXME: deal with STATE....

  // output
  const bool doOutput = true;
  if (doOutput) {

    // realm names
    NaluEnv::self().naluOutputP0()
      << "Xfer Setup Information: " << name_ << std::endl;
    NaluEnv::self().naluOutputP0()
      << "the From realm name is: " << fromRealm_->name_ << std::endl;
    NaluEnv::self().naluOutputP0()
      << "the To realm name is: " << toRealm_->name_ << std::endl;

    // provide mesh part names for the user
    NaluEnv::self().naluOutputP0() << "From/To Part Review: " << std::endl;
    for (size_t k = 0; k < fromPartVec_.size(); ++k)
      NaluEnv::self().naluOutputP0()
        << "the From mesh part name is: " << fromPartVec_[k]->name()
        << std::endl;
    for (size_t k = 0; k < toPartVec_.size(); ++k)
      NaluEnv::self().naluOutputP0()
        << "the To mesh part name is: " << toPartVec_[k]->name() << std::endl;

    // provide field names
    for (std::vector<std::pair<std::string, std::string>>::const_iterator
           i_var = transferVariablesPairName_.begin();
         i_var != transferVariablesPairName_.end(); ++i_var) {
      const std::pair<std::string, std::string> thePair = *i_var;
      NaluEnv::self().naluOutputP0()
        << "From variable " << thePair.first << " To variable "
        << thePair.second << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- allocate_stk_transfer -------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::allocate_stk_transfer()
{

  const stk::mesh::MetaData& fromMetaData = fromRealm_->meta_data();
  stk::mesh::BulkData& fromBulkData = fromRealm_->bulk_data();
  const std::string& fromcoordName = fromRealm_->get_coordinates_name();
  const std::vector<std::pair<std::string, std::string>>& FromVar =
    transferVariablesPairName_;
  const stk::ParallelMachine& fromComm = fromRealm_->bulk_data().parallel();

  std::shared_ptr<FromMesh> from_mesh(new FromMesh(
    fromMetaData, fromBulkData, *fromRealm_, fromcoordName, FromVar,
    fromPartVec_, fromComm));

  stk::mesh::MetaData& toMetaData = toRealm_->meta_data();
  stk::mesh::BulkData& toBulkData = toRealm_->bulk_data();
  const std::string& tocoordName = toRealm_->get_coordinates_name();
  const std::vector<std::pair<std::string, std::string>>& toVar =
    transferVariablesPairName_;
  const stk::ParallelMachine& toComm = toRealm_->bulk_data().parallel();

  std::shared_ptr<ToMesh> to_mesh(new ToMesh(
    toMetaData, toBulkData, *toRealm_, tocoordName, toVar, toPartVec_, toComm,
    searchTolerance_));

  typedef stk::transfer::GeometricTransfer<
    class LinInterp<class FromMesh, class ToMesh>>
    STKTransfer;

  // extract search type
  stk::search::SearchMethod searchMethod = stk::search::KDTREE;
  if (searchMethodName_ == "boost_rtree") {
    searchMethod = stk::search::KDTREE;
    NaluEnv::self().naluOutputP0()
      << "Warning: search method 'boost_rtree' has been deprecated"
      << ", switching to 'stk_kdtree'" << std::endl;
  } else if (searchMethodName_ == "stk_kdtree")
    searchMethod = stk::search::KDTREE;
  else
    NaluEnv::self().naluOutputP0()
      << "Transfer::search method not declared; will use stk_kdtree"
      << std::endl;
  transfer_.reset(new STKTransfer(
    from_mesh, to_mesh, name_, searchExpansionFactor_, searchMethod));
}

//--------------------------------------------------------------------------
//-------- ghost_from_elements ---------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::ghost_from_elements()
{
  typedef stk::transfer::GeometricTransfer<
    class LinInterp<class FromMesh, class ToMesh>>
    STKTransfer;

  const std::shared_ptr<STKTransfer> transfer =
    std::dynamic_pointer_cast<STKTransfer>(transfer_);
  const std::shared_ptr<STKTransfer::MeshA> mesha = transfer->meshA();

  STKTransfer::MeshA::EntityProcVec entity_keys;
  transfer->determine_entities_to_copy(entity_keys);
  mesha->update_ghosting(entity_keys);
}

//--------------------------------------------------------------------------
//-------- initialize_begin
//------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::initialize_begin()
{
  NaluEnv::self().naluOutputP0()
    << "PROCESSING Transfer::initialize_begin() for: " << name_ << std::endl;
  double time = -NaluEnv::self().nalu_time();
  allocate_stk_transfer();
  transfer_->coarse_search();
  time += NaluEnv::self().nalu_time();
  fromRealm_->timerTransferSearch_ += time;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::change_ghosting()
{
  ghost_from_elements();
}

//--------------------------------------------------------------------------
//-------- initialize_end ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::initialize_end()
{
  NaluEnv::self().naluOutputP0()
    << "PROCESSING Transfer::initialize_end() for: " << name_ << std::endl;
  transfer_->local_search();
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::execute()
{
  // do the xfer
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0()
    << "PROCESSING Transfer::execute() for: " << name_ << std::endl;

  // provide field names
  for (std::vector<std::pair<std::string, std::string>>::const_iterator i_var =
         transferVariablesPairName_.begin();
       i_var != transferVariablesPairName_.end(); ++i_var) {
    const std::pair<std::string, std::string> thePair = *i_var;
    NaluEnv::self().naluOutputP0()
      << "XFER From variable: " << thePair.first << " To variable "
      << thePair.second << std::endl;
  }
  NaluEnv::self().naluOutputP0() << std::endl;
  transfer_->apply();
}

Simulation*
Transfer::root()
{
  return parent()->root();
}
Transfers*
Transfer::parent()
{
  return &transfers_;
}

} // namespace nalu
} // namespace sierra
