// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#ifndef UTILITIESACTUATOR_H_
#define UTILITIESACTUATOR_H_

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

struct Coordinates;

namespace actuator_utils {

template<typename T>
inline
void reduce_view_on_host(T view){
  ThrowAssert(view.size()>0);
  ThrowAssert(view.data());
  MPI_Allreduce(
    MPI_IN_PLACE,
    view.data(),
    view.size(),
    MPI_DOUBLE, // TODO can we get this from the view?
    MPI_SUM,
    NaluEnv::self().parallel_comm());
}

// A Gaussian projection function
double Gaussian_projection(
  int nDim,
  double *dis,
  const Coordinates &epsilon);

// A Gaussian projection function
double Gaussian_projection(
  int nDim,
  double *dis,
  double *epsilon);

void resize_std_vector(
  const int& sizeOfField,
  std::vector<double>& theVector,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData);

void gather_field(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement);

void gather_field_for_interp(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement);

void interpolate_field(
  const int& sizeOfField,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData,
  const double* isoParCoords,
  const double* fieldAtNodes,
  double* pointField);

void
compute_distance(
  int nDim,
  const double *elemCentroid,
  const double *pointCentroid,
  double *distance);
}  // namespace actuator_utils
}  // namespace actuator_utils
}  // namespace actuator_utils

#endif
