// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <aero/AeroContainer.h>
#include <NaluEnv.h>
#include <NaluParsingHelper.h>
#ifdef NALU_USES_OPENFAST_FSI
#include "aero/fsi/OpenfastFSI.h"
#endif
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {
void
AeroContainer::clean_up()
{
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi())
    fsiContainer_->end_openfast();
#endif
}

AeroContainer::~AeroContainer()
{
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    delete fsiContainer_;
  }
#endif
}

AeroContainer::AeroContainer(const YAML::Node& node) : fsiContainer_(nullptr)
{
  // look for Actuator
  std::vector<const YAML::Node*> foundActuator;
  NaluParsingHelper::find_nodes_given_key("actuator", node, foundActuator);
  if (foundActuator.size() > 0) {
    if (foundActuator.size() != 1)
      throw std::runtime_error(
        "look_ahead_and_create::error: Too many actuator line blocks");
    actuatorModel_.parse(*foundActuator[0]);
  }
  // std::vector<const YAML::Node*> foundFsi;
  // NaluParsingHelper::find_nodes_given_key("openfast_fsi", node, foundFsi);
  if (node["openfast_fsi"]) {
#ifdef NALU_USES_OPENFAST_FSI
    // if (foundFsi.size() != 1)
    //   throw std::runtime_error(
    //     "look_ahead_and_create::error: Too many openfast_fsi blocks");
    fsiContainer_ = new OpenfastFSI(node["openfast_fsi"]);
#else
    throw std::runtime_error(
      "FSI can not be used without a specialized branch of openfast yet");
#endif
  }
}

void
AeroContainer::register_nodal_fields(
  stk::mesh::MetaData& meta, const stk::mesh::PartVector& part_vec)
{
  if (has_actuators()) {
    stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);
    const int nDim = meta.spatial_dimension();
    VectorFieldType* actuatorSource = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source"));
    VectorFieldType* actuatorSourceLHS = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source_lhs"));
    stk::mesh::put_field_on_mesh(*actuatorSource, selector, nDim, nullptr);
    stk::mesh::put_field_on_mesh(*actuatorSourceLHS, selector, nDim, nullptr);
  }
}

void
AeroContainer::setup(double timeStep, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  if (has_actuators()) {
    actuatorModel_.setup(timeStep, *bulk_);
  }
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    fsiContainer_->setup(timeStep, bulk_);
  }
#endif
}

void
AeroContainer::init(double currentTime, double restartFrequency)
{
  if (has_actuators()) {
    actuatorModel_.init(*bulk_);
  }
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    fsiContainer_->initialize(restartFrequency, currentTime);
  }
#else
  (void)currentTime;
  (void)restartFrequency;
#endif
}

void
AeroContainer::execute(double& actTimer)
{
  if (has_actuators()) {
    actuatorModel_.execute(actTimer);
  }
}
void
AeroContainer::update_displacements(const double currentTime, bool updateCC, bool predict)
{
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    NaluEnv::self().naluOutputP0() << "Calling update displacements inside AeroContainer" << std::endl;
    if (predict)
      fsiContainer_->predict_struct_states();
    fsiContainer_->map_displacements(currentTime, updateCC);
  }
#else
  (void)currentTime;
#endif
}

void
AeroContainer::predict_model_time_step(const double currentTime)
{
  (void)currentTime;
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    fsiContainer_->predict_struct_timestep(currentTime);
  }
#else
  (void)currentTime;
#endif
}

void
AeroContainer::advance_model_time_step(const double currentTime)
{
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    fsiContainer_->advance_struct_timestep(currentTime);
  }
#else
  (void)currentTime;
#endif
}

void
AeroContainer::compute_div_mesh_velocity()
{
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    fsiContainer_->compute_div_mesh_velocity();
  }
#endif
}

const stk::mesh::PartVector
AeroContainer::fsi_parts()
{
  stk::mesh::PartVector all_part_vec;
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    auto n_turbines = fsiContainer_->get_nTurbinesGlob();
    for (auto i_turb = 0; i_turb < n_turbines; i_turb++) {
      auto part_vec = fsiContainer_->get_fsiTurbineData(i_turb)->getPartVec();
      for (auto* part : part_vec)
        all_part_vec.push_back(part);
    }
  }
#endif
  return all_part_vec;
}

const stk::mesh::PartVector
AeroContainer::fsi_bndry_parts()
{
  stk::mesh::PartVector all_bndry_part_vec;
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    auto n_turbines = fsiContainer_->get_nTurbinesGlob();
    for (auto i_turb = 0; i_turb < n_turbines; i_turb++) {
      auto part_vec =
        fsiContainer_->get_fsiTurbineData(i_turb)->getBndryPartVec();
      for (auto* part : part_vec)
        all_bndry_part_vec.push_back(part);
    }
  }
#endif
  return all_bndry_part_vec;
}

const std::vector<std::string>
AeroContainer::fsi_bndry_part_names()
{
  std::vector<std::string> bndry_part_names;
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi()) {
    auto n_turbines = fsiContainer_->get_nTurbinesGlob();
    for (auto i_turb = 0; i_turb < n_turbines; i_turb++) {
      auto bp_names =
        fsiContainer_->get_fsiTurbineData(i_turb)->getBndryPartNames();
      for (auto bp : bp_names)
        bndry_part_names.push_back(bp);
    }
  }
#endif
  return bndry_part_names;
}

double
AeroContainer::openfast_accumulated_time()
{
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi())
    return fsiContainer_->total_openfastfsi_execution_time();
  else
    return -1.0;
#endif
  return -1.0;
}

double
AeroContainer::nalu_fsi_accumulated_time()
{
#ifdef NALU_USES_OPENFAST_FSI
  if (has_fsi())
    return fsiContainer_->total_nalu_fsi_execution_time();
  else
    return -1.0;
#endif
  return -1.0;
}

} // namespace nalu
} // namespace sierra
