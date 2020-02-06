// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorSearch.h>
#include <stk_search/CoarseSearch.hpp>
#include <FieldTypeDef.h>
#include <NaluEnv.h>

namespace sierra{
namespace nalu{

VecBoundSphere CreateBoundingSpheres(ActFixVectorDbl points, ActFixScalarDbl radius){

  const int nPoints = points.extent(0);
  // TODO(psakiev) can this be pre-allocated and modify entry values?
  // Will need to recreate every timestep with actuator line
  VecBoundSphere boundSphereVec;

  for(int i =0 ; i< nPoints; i++){
    // ID is zero bc we are only doing a local search (COMM_SELF)
    stk::search::IdentProc<uint64_t, int> theIdent((std::size_t)i, 0);

    Point thePoint (points(i,0), points(i,1), points(i,2));

    boundingSphere aSphere(Sphere(thePoint, radius(i)), theIdent);
    boundSphereVec.push_back(aSphere);
  }

  return boundSphereVec;
}

// refactor later
VecBoundElemBox CreateElementBoxes(stk::mesh::MetaData& stkMeta, stk::mesh::BulkData& stkBulk, std::vector<std::string> partNameList){
  VecBoundElemBox boundElemBoxVec;
  const int nDim = stkMeta.spatial_dimension();

  // fields
  VectorFieldType* coordinates = stkMeta.get_field<VectorFieldType>(
                                   stk::topology::NODE_RANK, "coordinates");

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

VecSearchKeyPair ExecuteCoarseSearch(
  VecBoundSphere& spheres,
  VecBoundElemBox& elems,
  stk::search::SearchMethod searchMethod)
{
  VecSearchKeyPair searchKeyPair;
  stk::search::coarse_search(spheres, elems, searchMethod, MPI_COMM_SELF, searchKeyPair);
  return searchKeyPair;
}

void ExecuteFineSearch(
  stk::mesh::MetaData& stkMeta,
  stk::mesh::BulkData& stkBulk,
  VecSearchKeyPair& coarseResults,
  ActFixElemIds matchElemIds
  )
{
  /*const int nDim = stkMeta.spatial_dimension();

  // extract fields
  VectorFieldType* coordinates = stkMeta.get_field<VectorFieldType>(
                                   stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // now proceed with the standard search
  std::vector<
  std::pair<boundingSphere::second_type, boundingElementBox::second_type>>::
      const_iterator ii;
  for (ii = coarseResults.begin(); ii != coarseResults.end(); ++ii) {

    const uint64_t thePt = ii->first.id();
    const uint64_t theBox = ii->second.id();
    const unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();

    // check if I own the point...
    if (theRank == pt_proc) {

      // yes, I own the point...

      // proceed as required; all elements should have already been ghosted via
      // the coarse search
      stk::mesh::Entity elem =
        stkBulk.get_entity(stk::topology::ELEMENT_RANK, theBox);
      if (!(stkBulk.is_valid(elem)))
        throw std::runtime_error("no valid entry for element");

      // find the point data structure
      std::map<size_t, std::unique_ptr<ActuatorPointInfo>>::iterator iterPoint;
      iterPoint = actuatorPointInfoMap_.find(thePt);
      if (iterPoint == actuatorPointInfoMap_.end())
        throw std::runtime_error("no valid entry for actuatorPointInfoMap_");

      // extract the point object and push back the element to either the best
      // candidate or the standard vector of elements
      ActuatorPointInfo* actuatorPointInfo = iterPoint->second.get();

      // extract topo and master element for this topo
      const stk::mesh::Bucket& theBucket = stkBulk.bucket(elem);
      const stk::topology& elemTopo = theBucket.topology();
      MasterElement* meSCS =
        sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
      const int nodesPerElement = meSCS->nodesPerElement_;

      // gather elemental coords
      std::vector<double> elementCoords(nDim * nodesPerElement);
      gather_field_for_interp(
        nDim, &elementCoords[0], *coordinates, stkBulk.begin_nodes(elem),
        nodesPerElement);

      // find isoparametric points
      std::vector<double> isoParCoords(nDim);
      const double nearestDistance = meSCS->isInElement(
                                       &elementCoords[0], &(actuatorPointInfo->centroidCoords_[0]),
                                       &(isoParCoords[0]));

      // save off best element and its isoparametric coordinates for this point
      if (nearestDistance < actuatorPointInfo->bestX_) {
        actuatorPointInfo->bestX_ = nearestDistance;
        actuatorPointInfo->isoParCoords_ = isoParCoords;
        actuatorPointInfo->bestElem_ = elem;
      }
      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = stkBulk.begin_nodes(elem);
      const unsigned num_nodes = stkBulk.num_nodes(elem);
      for (unsigned inode = 0; inode < num_nodes; inode++) {
        stk::mesh::Entity node = elem_node_rels[inode];
        actuatorPointInfo->nodeVec_.insert(node);
      }*/
}

} //namespace nalu
} //namespace sierra
