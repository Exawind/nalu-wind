// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <FieldRegistry.h>
#include <stk_topology/topology.hpp>
#include <functional>

namespace sierra {
namespace nalu {

static const DefVectorUnstated StatedNodalVector = {stk::topology::NODE_RANK};
static const DefScalarUnstated StatedNodalScalar = {stk::topology::NODE_RANK};

static const std::map<std::string, FieldDefTypes> Registry = {
  {"velocity", StatedNodalVector},
  {"temperature", StatedNodalScalar},
};

FieldRegistry::FieldRegistry() : database_(Registry) {}

} // namespace nalu
} // namespace sierra
