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
template <int NUM_DIM, int NUM_STATES>
const std::map<std::string, FieldDefTypes>&
Registry()
{
  // clang-format off
  FieldDefVector MultiStateNodalVector = {stk::topology::NODE_RANK, NUM_STATES, NUM_DIM};
  FieldDefScalar MultiStateNodalScalar = {stk::topology::NODE_RANK, NUM_STATES};

  FieldDefVector SingleStateNodalVector = {stk::topology::NODE_RANK, 1, NUM_DIM};
  FieldDefTensor SingleStateNodalTensor = {stk::topology::NODE_RANK, 1, NUM_DIM*NUM_DIM};
  FieldDefVector SingleStateEdgeVector = {stk::topology::EDGE_RANK, 1, NUM_DIM};
  FieldDefScalar SingleStateNodalScalar = {stk::topology::NODE_RANK};
  FieldDefScalar SingleStateElemScalar = {stk::topology::ELEM_RANK};
  FieldDefGeneric SingleStateEdgeGeneric = {stk::topology::EDGE_RANK};

  FieldDefTpetraId TpetraId = {stk::topology::NODE_RANK};
  FieldDefGlobalId GlobalId = {stk::topology::NODE_RANK};
  FieldDefHypreId HypreId = {stk::topology::NODE_RANK};
  FieldDefScalarInt NodalScalarInt = {stk::topology::NODE_RANK};

  static const std::map<std::string, FieldDefTypes> registry = {
    {"velocity", MultiStateNodalVector},
    {"temperature", MultiStateNodalScalar},
    {"hypre_global_id", HypreId},
    {"tpet_global_id", TpetraId},
    {"nalu_global_id", GlobalId},
    {"dual_nodal_volume", SingleStateNodalScalar},
    {"element_volume", SingleStateElemScalar},
    {"edge_area_vector", SingleStateEdgeVector},
    {"mesh_displacement", MultiStateNodalVector},
    {"current_coordinates", SingleStateNodalVector},
    {"mesh_velocity", SingleStateNodalVector},
    {"velocity_rtm", SingleStateNodalVector},
    {"div_mesh_velocity", SingleStateNodalScalar},
    {"density", SingleStateNodalScalar},
    {"viscosity", SingleStateNodalScalar}, 
    {"turbulent_ke", SingleStateNodalScalar}, 
    {"specific_dissipation_rate", SingleStateNodalScalar}, 
    {"minimum_distance_to_wall", SingleStateNodalScalar}, 
    {"turbulent_viscosity" ,       SingleStateNodalScalar}, 
    {"sst_max_length_scale",       SingleStateNodalScalar}, 
    {"sst_f_one_blending"  ,       SingleStateNodalScalar}, 
    {"effective_viscosity" ,       SingleStateNodalScalar}, 
    {"specific_heat" ,       SingleStateNodalScalar}, 
    {"rans_time_scale" ,       SingleStateNodalScalar}, 
    {"open_tke_bc", SingleStateNodalScalar},
    {"dkdx", SingleStateNodalVector},
    {"dwdx", SingleStateNodalVector},
    {"dhdx", SingleStateNodalVector},
    {"dudx", SingleStateNodalTensor},
    {"average_dudx" ,       SingleStateNodalTensor}, 
    {"open_mass_flow_rate" ,       SingleStateEdgeGeneric}, 
    {"iblank", NodalScalarInt}
  };
  // clang-format on
  return registry;
}

FieldRegistry::FieldRegistry()
  : database_2D_2_state_(Registry<2, 2>()),
    database_2D_3_state_(Registry<2, 3>()),
    database_3D_2_state_(Registry<3, 2>()),
    database_3D_3_state_(Registry<3, 3>())
{
}

} // namespace nalu
} // namespace sierra
