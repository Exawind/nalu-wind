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
  FieldDefGeneric SingleStateElemGeneric = FieldDefGeneric{stk::topology::ELEM_RANK};
  FieldDefGeneric SingleStateEdgeGeneric = FieldDefGeneric{stk::topology::EDGE_RANK};
  FieldDefGeneric SingleStateNodeGeneric = FieldDefGeneric{stk::topology::NODE_RANK};

  FieldDefScalar  SingleStateNodalScalar = FieldDefScalar{stk::topology::NODE_RANK};
  FieldDefScalar  SingleStateElemScalar  = FieldDefScalar{stk::topology::ELEM_RANK};
  FieldDefVector  SingleStateElemVector  = FieldDefVector{stk::topology::ELEM_RANK, 1, NUM_DIM};
  FieldDefScalar  MultiStateNodalScalar  = FieldDefScalar{stk::topology::NODE_RANK, NUM_STATES};

  FieldDefVector  SingleStateNodalVector = FieldDefVector{stk::topology::NODE_RANK, 1, NUM_DIM};
  FieldDefVector  SingleStateEdgeVector  = FieldDefVector{stk::topology::EDGE_RANK, 1, NUM_DIM};
  FieldDefVector  MultiStateNodalVector  = FieldDefVector{stk::topology::NODE_RANK, NUM_STATES, NUM_DIM};

  FieldDefTensor  SingleStateNodalTensor = FieldDefTensor{stk::topology::NODE_RANK, 1, NUM_DIM*NUM_DIM};

  FieldDefTpetraId  TpetraId             = FieldDefTpetraId{stk::topology::NODE_RANK};
  FieldDefGlobalId  GlobalId             = FieldDefGlobalId{stk::topology::NODE_RANK};
  FieldDefHypreId   HypreId              = FieldDefHypreId{stk::topology::NODE_RANK};
  FieldDefScalarInt NodalScalarInt       = FieldDefScalarInt{stk::topology::NODE_RANK};

  static const std::map<std::string, FieldDefTypes> registry = {
    {"average_dudx" ,             SingleStateNodalTensor},
    {"coordinates",               SingleStateNodalVector},
    {"coordinates_copy",          SingleStateNodalVector}, // Used in testing
    {"current_coordinates",       SingleStateNodalVector},
    {"density",                   SingleStateNodalScalar},
    {"dhdx",                      SingleStateNodalVector},
    {"diffFluxCoeff",             SingleStateNodalScalar},
    {"discreteLaplacian",         SingleStateNodalScalar},
    {"div_mesh_velocity",         SingleStateNodalScalar},
    {"dkdx",                      SingleStateNodalVector},
    {"dpdx",                      SingleStateNodalVector},
    {"dual_nodal_volume",         SingleStateNodalScalar},
    {"dudx",                      SingleStateNodalTensor},
    {"dwdx",                      SingleStateNodalVector},
    {"edge_area_vector",          SingleStateEdgeVector},
    {"effective_viscosity" ,      SingleStateNodalScalar},
    {"elemCentroid",              SingleStateElemVector},
    {"element_volume",            SingleStateElemScalar},
    {"elemScalarField",           SingleStateElemScalar},  // Used in testing
    {"elemTensorField",           SingleStateElemGeneric}, // Used in testing
    {"elemVectorField",           SingleStateElemGeneric}, // Used in testing
    {"exposed_area_vector",       SingleStateEdgeGeneric},
    {"Gju",                       SingleStateNodeGeneric},
    {"hypre_global_id",           HypreId},
    {"iblank",                    NodalScalarInt},
    {"idField",                   SingleStateNodalScalar},
    {"mass_flow_rate_scs",        SingleStateElemGeneric},
    {"mesh_displacement",         MultiStateNodalVector},
    {"mesh_velocity",             SingleStateNodalVector},
    {"minimum_distance_to_wall",  SingleStateNodalScalar},
    {"momentum_diag",             SingleStateNodalScalar},
    {"nalu_global_id",            GlobalId},
    {"nodalScalarField",          SingleStateNodalScalar}, // Used in testing
    {"nodalGenericField",         SingleStateNodeGeneric}, // Used in testing
    {"nodalPressure",             SingleStateNodalScalar},
    {"nodalTensorField",          SingleStateNodalTensor}, // Used in testing
    {"open_mass_flow_rate" ,      SingleStateEdgeGeneric},
    {"open_tke_bc",               SingleStateNodalScalar},
    {"pressure",                  SingleStateNodalScalar},
    {"rans_time_scale" ,          SingleStateNodalScalar},
    {"scalarQ",                   SingleStateNodalScalar},
    {"specific_dissipation_rate", MultiStateNodalScalar},
    {"specific_heat" ,            SingleStateNodalScalar},
    {"sst_f_one_blending"  ,      SingleStateNodalScalar},
    {"sst_max_length_scale",      SingleStateNodalScalar},
    {"temperature",               MultiStateNodalScalar},
    {"tpet_global_id",            TpetraId},
    {"turbulent_ke",              SingleStateNodalScalar},
    {"turbulent_viscosity" ,      SingleStateNodalScalar},
    {"velocity",                  MultiStateNodalVector},
    {"velocity_rtm",              SingleStateNodalVector},
    {"viscosity",                 SingleStateNodalScalar}
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
