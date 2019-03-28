/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <Actuator.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>

// master elements
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>


// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// basic c++
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>

namespace sierra {
namespace nalu {

Actuator::Actuator(Realm& realm, const YAML::Node& node)
  : realm_(realm),
    searchMethod_(stk::search::KDTREE),
    needToGhostCount_(0),
    actuatorGhosting_(NULL)
{
  load(node);
}

Actuator::~Actuator() {}

void
Actuator::load(const YAML::Node& y_node)
{
  // check for any data probes
  const YAML::Node y_actuator = y_node["actuator"];
  if (y_actuator) {
    NaluEnv::self().naluOutputP0() << "Actuator::load" << std::endl;

    // search specifications
    std::string searchMethodName = "na";
    get_if_present(
      y_actuator, "search_method", searchMethodName, searchMethodName);

    // determine search method for this pair
    if (searchMethodName == "boost_rtree")
      searchMethod_ = stk::search::BOOST_RTREE;
    else if (searchMethodName == "stk_kdtree")
      searchMethod_ = stk::search::KDTREE;
    else
      NaluEnv::self().naluOutputP0()
        << "Actuator::search method not declared; will use stk_kdtree"
        << std::endl;

    // extract the set of from target names; each spec is homogeneous in this
    // respect
    const YAML::Node searchTargets = y_actuator["search_target_part"];
    if (searchTargets.Type() == YAML::NodeType::Scalar) {
      searchTargetNames_.resize(1);
      searchTargetNames_[0] = searchTargets.as<std::string>();
    } else {
      searchTargetNames_.resize(searchTargets.size());
      for (size_t i = 0; i < searchTargets.size(); ++i) {
        searchTargetNames_[i] = searchTargets[i].as<std::string>();
      }
    }
  }
}

void
Actuator::populate_candidate_elements()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();
  stk::mesh::BulkData& bulkData = realm_.bulk_data();

  const int nDim = metaData.spatial_dimension();

  // fields
  VectorFieldType* coordinates = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // point data structures
  Point minCorner, maxCorner;

  // extract part
  stk::mesh::PartVector searchParts;
  for (size_t k = 0; k < searchTargetNames_.size(); ++k) {
    stk::mesh::Part* thePart = metaData.get_part(searchTargetNames_[k]);
    if (NULL != thePart)
      searchParts.push_back(thePart);
    else
      throw std::runtime_error(
        get_class_name() + ": Part is null" + searchTargetNames_[k]);
  }

  // selector and bucket loop
  stk::mesh::Selector s_locally_owned =
    metaData.locally_owned_part() & stk::mesh::selectUnion(searchParts);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned);

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
      stk::mesh::Entity const* elem_node_rels = bulkData.begin_nodes(elem);
      const int num_nodes = bulkData.num_nodes(elem);

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
      stk::search::IdentProc<uint64_t, int> theIdent(
        bulkData.identifier(elem), NaluEnv::self().parallel_rank());

      // create the bounding point box and push back
      boundingElementBox theBox(Box(minCorner, maxCorner), theIdent);
      boundingElementBoxVec_.push_back(theBox);
    }
  }
}

void
Actuator::manage_ghosting()
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();

  // check for ghosting need
  uint64_t g_needToGhostCount = 0;
  stk::all_reduce_sum(
    NaluEnv::self().parallel_comm(), &needToGhostCount_, &g_needToGhostCount,
    1);
  if (g_needToGhostCount > 0) {
    NaluEnv::self().naluOutputP0()
      << get_class_name() + " alg will ghost a number of entities: "
      << g_needToGhostCount << std::endl;
    bulkData.modification_begin();
    bulkData.change_ghosting(*actuatorGhosting_, elemsToGhost_);
    bulkData.modification_end();
  } else {
    NaluEnv::self().naluOutputP0()
      << get_class_name() + " alg will NOT ghost entities: " << std::endl;
  }
}

void
Actuator::complete_search()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  const int nDim = metaData.spatial_dimension();

  // extract fields
  VectorFieldType* coordinates = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // now proceed with the standard search
  std::vector<
    std::pair<boundingSphere::second_type, boundingElementBox::second_type>>::
    const_iterator ii;
  for (ii = searchKeyPair_.begin(); ii != searchKeyPair_.end(); ++ii) {

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
        bulkData.get_entity(stk::topology::ELEMENT_RANK, theBox);
      if (!(bulkData.is_valid(elem)))
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
      const stk::mesh::Bucket& theBucket = bulkData.bucket(elem);
      const stk::topology& elemTopo = theBucket.topology();
      MasterElement* meSCS =
        sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
      const int nodesPerElement = meSCS->nodesPerElement_;

      // gather elemental coords
      std::vector<double> elementCoords(nDim * nodesPerElement);
      gather_field_for_interp(
        nDim, &elementCoords[0], *coordinates, bulkData.begin_nodes(elem),
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
      stk::mesh::Entity const* elem_node_rels = bulkData.begin_nodes(elem);
      const unsigned num_nodes = bulkData.num_nodes(elem);
      for (unsigned inode = 0; inode < num_nodes; inode++) {
        stk::mesh::Entity node = elem_node_rels[inode];
        actuatorPointInfo->nodeVec_.insert(node);
      }
    } else {
      // not this proc's issue
    }
  }
}

void
Actuator::determine_elems_to_ghost()
{
  stk::mesh::BulkData& bulkData = realm_.bulk_data();

  stk::search::coarse_search(
    boundingSphereVec_, boundingElementBoxVec_, searchMethod_,
    NaluEnv::self().parallel_comm(), searchKeyPair_);

  // lowest effort is to ghost elements to the owning rank of the point; can
  // just as easily do the opposite
  std::vector<
    std::pair<boundingSphere::second_type, boundingElementBox::second_type>>::
    const_iterator ii;
  for (ii = searchKeyPair_.begin(); ii != searchKeyPair_.end(); ++ii) {

    const uint64_t theBox = ii->second.id();
    unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();
    const unsigned box_proc = ii->second.proc();
    if ((box_proc == theRank) && (pt_proc != theRank)) {

      // Send box to pt proc

      // find the element
      stk::mesh::Entity theElemMeshObj =
        bulkData.get_entity(stk::topology::ELEMENT_RANK, theBox);
      if (!(bulkData.is_valid(theElemMeshObj)))
        throw std::runtime_error("no valid entry for element");

      // new element to ghost counter
      needToGhostCount_++;

      // deal with elements to push back to be ghosted
      stk::mesh::EntityProc theElemPair(theElemMeshObj, pt_proc);
      elemsToGhost_.push_back(theElemPair);
    }
  }
}

void
Actuator::resize_std_vector(
  const int& sizeOfField,
  std::vector<double>& theVector,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData)
{
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
  const int nodesPerElement = meSCS->nodesPerElement_;
  theVector.resize(nodesPerElement * sizeOfField);
}

//--------------------------------------------------------------------------
//-------- gather_field ----------------------------------------------------
//--------------------------------------------------------------------------
void
Actuator::gather_field(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement)
{
  for (int ni = 0; ni < nodesPerElement; ++ni) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double* theField = (double*)stk::mesh::field_data(stkField, node);
    for (int j = 0; j < sizeOfField; ++j) {
      const int offSet = ni * sizeOfField + j;
      fieldToFill[offSet] = theField[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- gather_field_for_interp -----------------------------------------
//--------------------------------------------------------------------------
void
Actuator::gather_field_for_interp(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement)
{
  for (int ni = 0; ni < nodesPerElement; ++ni) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double* theField = (double*)stk::mesh::field_data(stkField, node);
    for (int j = 0; j < sizeOfField; ++j) {
      const int offSet = j * nodesPerElement + ni;
      fieldToFill[offSet] = theField[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_volume --------------------------------------------------
//--------------------------------------------------------------------------
double
Actuator::compute_volume(
  const int & /* nDim */,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData & bulkData)
{
  // extract master element from the bucket in which the element resides
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);
  const int numScvIp = meSCV->num_integration_points();

  // compute scv for this element
  ws_scv_volume_.resize(numScvIp);
  double scv_error = 0.0;
  meSCV->determinant(1, &ws_coordinates_[0], &ws_scv_volume_[0], &scv_error);

  double elemVolume = 0.0;
  for (int ip = 0; ip < numScvIp; ++ip) {
    elemVolume += ws_scv_volume_[ip];
  }
  return elemVolume;
}

//--------------------------------------------------------------------------
//-------- interpolate_field -----------------------------------------------
//--------------------------------------------------------------------------
void
Actuator::interpolate_field(
  const int& sizeOfField,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData,
  const double* isoParCoords,
  const double* fieldAtNodes,
  double* pointField)
{
  // extract master element from the bucket in which the element resides
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);

  // interpolate velocity to this best point
  meSCS->interpolatePoint(sizeOfField, isoParCoords, fieldAtNodes, pointField);
}
double
Actuator::compute_distance(
  const int& nDim, const double* elemCentroid, const double* pointCentroid)
{
  double distance = 0.0;
  for (int j = 0; j < nDim; ++j)
    distance += std::pow(elemCentroid[j] - pointCentroid[j], 2);
  distance = std::sqrt(distance);
  return distance;
}

} // namespace nalu
} // namespace sierra
