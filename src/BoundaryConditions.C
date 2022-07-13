// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Realm.h>
#include <BoundaryConditions.h>
#include <NaluEnv.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// BoundaryCondition - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//-------- load -----------------------------------------------
//--------------------------------------------------------------------------

template <typename T>
std::unique_ptr<BoundaryCondition>
register_bc(const YAML::Node& node, BoundaryConditions& bcs)
{
  std::unique_ptr<BoundaryCondition> this_bc = std::make_unique<T>(bcs);
  auto* cast_bc = dynamic_cast<T*>(this_bc.get());
  node >> *cast_bc;
  return this_bc;
}

std::unique_ptr<BoundaryCondition>
BoundaryCondition::load(const YAML::Node& node)
{
  std::unique_ptr<BoundaryCondition> this_bc;
  if (node["wall_boundary_condition"]) {
    this_bc = std::move(
      register_bc<WallBoundaryConditionData>(node, boundaryConditions_));

    NaluEnv::self().naluOutputP0()
      << "Wall BC name:        " << this_bc->bcName_ << " on "
      << this_bc->targetName_ << std::endl;

  } else if (node["inflow_boundary_condition"]) {
    this_bc = std::move(
      register_bc<InflowBoundaryConditionData>(node, boundaryConditions_));

    NaluEnv::self().naluOutputP0()
      << "Inflow BC name:        " << this_bc->bcName_ << " on "
      << this_bc->targetName_ << std::endl;

  } else if (node["open_boundary_condition"]) {
    this_bc = std::move(
      register_bc<OpenBoundaryConditionData>(node, boundaryConditions_));

    NaluEnv::self().naluOutputP0()
      << "Open BC name:        " << this_bc->bcName_ << " on "
      << this_bc->targetName_ << std::endl;

  } else if (node["symmetry_boundary_condition"]) {
    this_bc = std::move(
      register_bc<SymmetryBoundaryConditionData>(node, boundaryConditions_));

    NaluEnv::self().naluOutputP0()
      << "Symmetry BC name:        " << this_bc->bcName_ << " on "
      << this_bc->targetName_ << std::endl;

  } else if (node["abltop_boundary_condition"]) {
    this_bc = std::move(
      register_bc<ABLTopBoundaryConditionData>(node, boundaryConditions_));

    NaluEnv::self().naluOutputP0()
      << "ABLTop BC name:        " << this_bc->bcName_ << " on "
      << this_bc->targetName_ << std::endl;

  } else if (node["periodic_boundary_condition"]) {
    this_bc = std::move(
      register_bc<PeriodicBoundaryConditionData>(node, boundaryConditions_));

    auto* periodicBC =
      dynamic_cast<PeriodicBoundaryConditionData*>(this_bc.get());

    NaluEnv::self().naluOutputP0()
      << "Periodic BC name:    " << periodicBC->bcName_ << " between "
      << periodicBC->masterSlave_.master_ << " and "
      << periodicBC->masterSlave_.slave_ << std::endl;

  } else if (node["non_conformal_boundary_condition"]) {
    this_bc = std::move(register_bc<NonConformalBoundaryConditionData>(
      node, boundaryConditions_));

    NaluEnv::self().naluOutputP0()
      << "NonConformal BC name:    " << this_bc->bcName_ << " using "
      << this_bc->targetName_ << std::endl;

  } else if (node["overset_boundary_condition"]) {
    this_bc = std::move(
      register_bc<OversetBoundaryConditionData>(node, boundaryConditions_));

    NaluEnv::self().naluOutputP0()
      << "Overset BC name: " << this_bc->bcName_ << std::endl;

  } else {
    throw std::runtime_error(
      "parser error BoundaryConditions::load: no such bc type");
  }
  return this_bc;
}

void
BoundaryConditions::load(const YAML::Node& node)
{
  BoundaryCondition bc_factory(*this);

  if (node["boundary_conditions"]) {
    const YAML::Node boundary_conditions = node["boundary_conditions"];

    for (auto&& bc_node : boundary_conditions) {
      boundaryConditionVector_.emplace_back(bc_factory.load(bc_node));
    }

  } else {
    throw std::runtime_error("parser error BoundaryConditions::load");
  }
}

} // namespace nalu
} // namespace sierra
