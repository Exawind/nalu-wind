// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorSearch.h>
#include <stk_search/CoarseSearch.hpp>
#include <FieldTypeDef.h>
#include <NaluEnv.h>
#include <aero/actuator/UtilitiesActuator.h>

namespace sierra {
namespace nalu {

VecBoundSphere
CreateBoundingSpheres(ActFixVectorDbl points, ActFixScalarDbl radius)
{

  const int nPoints = points.extent(0);
  // TODO can this be pre-allocated and modify entry values?
  // Will need to recreate every timestep with actuator line
  VecBoundSphere boundSphereVec;

  for (int i = 0; i < nPoints; i++) {
    // ID is zero bc we are only doing a local search (COMM_SELF)
    stk::search::IdentProc<uint64_t, int> theIdent((std::size_t)i, 0);

    Point thePoint(points(i, 0), points(i, 1), points(i, 2));

    boundingSphere aSphere(Sphere(thePoint, radius(i)), theIdent);
    boundSphereVec.push_back(aSphere);
  }

  return boundSphereVec;
}

// refactor later
VecBoundElemBox
CreateElementBoxes(
  stk::mesh::BulkData& stkBulk, std::vector<std::string> partNameList)
{
  VecBoundElemBox boundElemBoxVec;
  const int nDim = 3;
  stk::mesh::MetaData& stkMeta = stkBulk.mesh_meta_data();

  // fields
  VectorFieldType* coordinates =
    stkMeta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  // point data structures
  Point minCorner, maxCorner;

  // extract part
  stk::mesh::PartVector searchParts;
  for (size_t k = 0; k < partNameList.size(); ++k) {
    stk::mesh::Part* thePart = stkMeta.get_part(partNameList[k]);
    if (NULL != thePart)
      searchParts.push_back(thePart);
    else
      throw std::runtime_error(
        "ActuatorSearch::CreateElemenBoxes: Part is null" + partNameList[k]);
  }

  // selector and bucket loop
  stk::mesh::Selector s_locally_owned =
    stkMeta.locally_owned_part() & stk::mesh::selectUnion(searchParts);

  stk::mesh::BucketVector const& elem_buckets =
    stkBulk.get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned);

  for (stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
       ib != elem_buckets.end(); ++ib) {

    stk::mesh::Bucket& b = **ib;

    const stk::mesh::Bucket::size_type length = b.size();

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      // get element
      stk::mesh::Entity elem = b[k];

      // initialize max and min
      for (int j = 0; j < nDim; ++j) {
        minCorner[j] = +1.0e16;
        maxCorner[j] = -1.0e16;
      }

      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = stkBulk.begin_nodes(elem);
      const int num_nodes = stkBulk.num_nodes(elem);

      for (int ni = 0; ni < num_nodes; ++ni) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // pointers to real data
        const double* coords = stk::mesh::field_data(*coordinates, node);

        // check max/min
        for (int j = 0; j < nDim; ++j) {
          minCorner[j] = std::min(minCorner[j], coords[j]);
          maxCorner[j] = std::max(maxCorner[j], coords[j]);
        }
      }

      // setup ident
      // ID is zero bc we are only doing a local search (COMM_SELF)
      stk::search::IdentProc<uint64_t, int> theIdent(
        stkBulk.identifier(elem), 0);

      // create the bounding point box and push back
      boundingElementBox theBox(Box(minCorner, maxCorner), theIdent);
      boundElemBoxVec.push_back(theBox);
    }
  }
  return boundElemBoxVec;
}

void
ExecuteCoarseSearch(
  VecBoundSphere& spheres,
  VecBoundElemBox& elems,
  ActScalarU64Dv& coarsePointIds,
  ActScalarU64Dv& coarseElemIds,
  stk::search::SearchMethod searchMethod)
{
  VecSearchKeyPair searchKeyPair;
  stk::search::coarse_search(
    spheres, elems, searchMethod, MPI_COMM_SELF, searchKeyPair);

  const std::size_t numLocalMatches = searchKeyPair.size();

  coarsePointIds.resize(numLocalMatches);
  coarseElemIds.resize(numLocalMatches);

  coarsePointIds.sync_host();
  coarseElemIds.sync_host();
  coarsePointIds.modify_host();
  coarseElemIds.modify_host();

  for (std::size_t i = 0; i < numLocalMatches; i++) {
    coarsePointIds.h_view(i) = searchKeyPair[i].first.id();
    coarseElemIds.h_view(i) = searchKeyPair[i].second.id();
  }
}

void
ExecuteFineSearch(
  stk::mesh::BulkData& stkBulk,
  ActScalarU64Dv coarsePointIds,
  ActScalarU64Dv coarseElemIds,
  ActFixVectorDbl points,
  ActFixElemIds matchElemIds,
  ActFixVectorDbl localCoords,
  ActFixScalarBool isLocalPoint,
  ActFixScalarInt localParallelRedundancy)
{
  const int nDim = 3;

  ThrowAssert(isLocalPoint.extent(0) == points.extent(0));
  ThrowAssert(coarsePointIds.extent(0) == coarseElemIds.extent(0));

  // extract fields
  stk::mesh::MetaData& stkMeta = stkBulk.mesh_meta_data();
  VectorFieldType* coordinates =
    stkMeta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  for (unsigned i = 0; i < isLocalPoint.extent(0); i++) {
    isLocalPoint(i) = false;
    localParallelRedundancy(i) = 0.0;
  }

  // now proceed with the standard search
  for (unsigned i = 0; i < coarseElemIds.extent(0); i++) {

    const uint64_t thePt = coarsePointIds.h_view(i);
    const uint64_t theBox = coarseElemIds.h_view(i);

    auto pointCoords = Kokkos::subview(points, thePt, Kokkos::ALL);
    auto localPntCrds = Kokkos::subview(localCoords, thePt, Kokkos::ALL);

    // all elements should be local bc of the coarse search
    stk::mesh::Entity elem =
      stkBulk.get_entity(stk::topology::ELEMENT_RANK, theBox);
    if (!(stkBulk.is_valid(elem)))
      throw std::runtime_error(
        "ExecuteFineSearch:: no valid entry for element");

    // extract topo and master element for this topo
    const stk::mesh::Bucket& theBucket = stkBulk.bucket(elem);
    const stk::topology& elemTopo = theBucket.topology();
    MasterElement* meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
    const int nodesPerElement = meSCS->nodesPerElement_;

    // gather elemental coords
    std::vector<double> elementCoords(nDim * nodesPerElement);
    actuator_utils::gather_field_for_interp(
      nDim, &elementCoords[0], *coordinates, stkBulk.begin_nodes(elem),
      nodesPerElement);

    // find isoparametric points
    std::vector<double> isoParCoords(nDim);
    const double nearestDistance = meSCS->isInElement(
      &elementCoords[0], pointCoords.data(), &(isoParCoords[0]));

    // if it is actually in the element save it
    if (std::abs(nearestDistance) <= 1.0) {
      matchElemIds(thePt) = theBox;
      isLocalPoint(thePt) = true;
      localParallelRedundancy(thePt) = 1.0;
      localPntCrds(0) = isoParCoords[0];
      localPntCrds(1) = isoParCoords[1];
      localPntCrds(2) = isoParCoords[2];
    }
  }
}

} // namespace nalu
} // namespace sierra
