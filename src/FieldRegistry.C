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

// Registry object is where all the fully quantified field definitions live
// This is the starting point for adding a new field
template <int NUM_STATES>
const std::map<std::string, FieldDefTypes>&
Registry()
{

  FieldDefVector MultiStateNodalVector = {stk::topology::NODE_RANK, NUM_STATES};
  FieldDefScalar MultiStateNodalScalar = {stk::topology::NODE_RANK, NUM_STATES};

  FieldDefVector SingleStateNodalVector = {stk::topology::NODE_RANK};
  FieldDefScalar SingleStateNodalScalar = {stk::topology::NODE_RANK};

  static const std::map<std::string, FieldDefTypes> registry = {
    {"velocity", MultiStateNodalVector},
    {"temperature", MultiStateNodalScalar},
  };
  return registry;
}

FieldRegistry::FieldRegistry()
  : database_2_state_(Registry<2>()), database_3_state_(Registry<3>())
{
}

} // namespace nalu
} // namespace sierra
