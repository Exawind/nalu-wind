// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <FieldRegistry.h>
#include <FieldStateLogic.h>
#include <stk_topology/topology.hpp>
#include <functional>

namespace sierra {
namespace nalu {
static const std::function<int(FieldStateLogic)> variable_states =
  [](FieldStateLogic l) { return l.useBDF2_ ? 3 : 2; };
static const std::function<int(FieldStateLogic)> constant_states =
  [](FieldStateLogic) { return 1; };

static const FieldDefinition StatedNodalVector = {
  stk::topology::NODE_RANK, FieldTypes::VECTOR, variable_states};
static const FieldDefinition StatedNodalScalar = {
  stk::topology::NODE_RANK, FieldTypes::SCALAR, variable_states};
static const FieldDefinition UnstatedNodalVector = {
  stk::topology::NODE_RANK, FieldTypes::VECTOR, constant_states};
static const FieldDefinition UnstatedNodalScalar = {
  stk::topology::NODE_RANK, FieldTypes::VECTOR, constant_states};

static const std::map<std::string, FieldDefinition> Registry = {
  {"velocity", StatedNodalVector},
  {"temperature", StatedNodalScalar},
  {"hypre_global_id",
   {stk::topology::NODE_RANK, FieldTypes::HYPREID, constant_states}},
  {"tpet_global_id",
   {stk::topology::NODE_RANK, FieldTypes::TPETID, constant_states}},
  {"nalu_global_id",
   {stk::topology::NODE_RANK, FieldTypes::GLOBALID, constant_states}},
};

FieldRegistry::FieldRegistry() : database_(Registry) {}

} // namespace nalu
} // namespace sierra
