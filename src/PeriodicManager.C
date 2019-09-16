/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <PeriodicManager.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <utils/StkHelpers.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/util/SortAndUnique.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// vector
#include <vector>
#include <map>
#include <string>

namespace sierra{
namespace nalu{

PeriodicManager::PeriodicManager(
   Realm &realm)
  : realm_(realm ),
    searchTolerance_(1.0e-8),
    periodicGhosting_(NULL),
    ghostingName_("nalu_periodic"),
    timerSearch_(0.0),
    errorCount_(0),
    maxErrorCount_(10),
    amplificationFactor_(5.0)
{
  // does nothing
}

PeriodicManager::~PeriodicManager()
{
  // nothing to delete
}

//--------------------------------------------------------------------------
//-------- initialize_error_count ------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::initialize_error_count()
{
  errorCount_ = 0;
}

//--------------------------------------------------------------------------
//-------- add periodic pair -----------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::add_periodic_pair(
  stk::mesh::Part * masterMeshPart,
  stk::mesh::Part * slaveMeshPart,
  const double &userSearchTolerance,
  const std::string &searchMethodName)
{
  // use most stringent tolerance (min) for all of user specifications
  searchTolerance_ = std::min(searchTolerance_, userSearchTolerance);

  // form the slave part vector
  slavePartVector_.push_back(slaveMeshPart);

  // form the selector pair
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::Selector masterSelect = meta_data.locally_owned_part() & stk::mesh::Selector(*masterMeshPart);
  stk::mesh::Selector slaveSelect = meta_data.locally_owned_part() & stk::mesh::Selector(*slaveMeshPart);
  SelectorPair periodicSelectorPair(masterSelect, slaveSelect);
  periodicSelectorPairs_.push_back(periodicSelectorPair);

  // determine search method for this pair; default is boost_rtree
  stk::search::SearchMethod searchMethod = stk::search::KDTREE;
  if ( searchMethodName == "boost_rtree" )
    searchMethod = stk::search::BOOST_RTREE;
  else if ( searchMethodName == "stk_kdtree" )
    searchMethod = stk::search::KDTREE;
  else
    NaluEnv::self().naluOutputP0() << "PeriodicManager::search method not declared; will use stk_kdtree" << std::endl;
  searchMethodVec_.push_back(searchMethod);
}

//--------------------------------------------------------------------------
//-------- get_slave_part_vector ----------------------------------------------
//--------------------------------------------------------------------------
const stk::mesh::PartVector &
PeriodicManager::get_slave_part_vector()
{
  return slavePartVector_;
}

//--------------------------------------------------------------------------
//-------- get_search_time -------------------------------------------------
//--------------------------------------------------------------------------
double
PeriodicManager::get_search_time() {
  return timerSearch_;
}

//--------------------------------------------------------------------------
//-------- build_constraints -----------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::build_constraints()
{
  if ( periodicSelectorPairs_.size() == 0 )
     throw std::runtime_error("PeriodiocBC::Error: No periodic pair provided");

  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "Periodic Review:  realm: " << realm_.name_ << std::endl;
  NaluEnv::self().naluOutputP0() << "=========================" << std::endl;

  augment_periodic_selector_pairs();

  // initialize translation and rotation vectors
  initialize_translation_vector();

  // translate
  for ( size_t k = 0; k < periodicSelectorPairs_.size(); ++k) {
    determine_translation(periodicSelectorPairs_[k].first, periodicSelectorPairs_[k].second,
        translationVector_[k], rotationVector_[k]);
  }

  remove_redundant_slave_nodes();

  // search and constraint mapping
  finalize_search();

  // provide Nalu id update
  update_global_id_field();
}

//--------------------------------------------------------------------------
//-------- augment_periodic_selector_pairs ---------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::augment_periodic_selector_pairs()
{
  
  const size_t pairSize = periodicSelectorPairs_.size();

  switch ( pairSize ) {

    case 1:
      break; // nothing to do

    case 2: {

      // master/slave selectors
      stk::mesh::Selector &masterA = periodicSelectorPairs_[0].first;
      stk::mesh::Selector &masterB = periodicSelectorPairs_[1].first;

      stk::mesh::Selector &slaveA = periodicSelectorPairs_[0].second;
      stk::mesh::Selector &slaveB = periodicSelectorPairs_[1].second;

      // push back intersection selector pairs
      periodicSelectorPairs_.push_back(std::make_pair(masterA & masterB, slaveA & slaveB));

      // need a search method; arbitrarily choose the first method specified
      stk::search::SearchMethod searchMethod = searchMethodVec_[0];
      searchMethodVec_.push_back(searchMethod);
    
      break;
    }

    case 3: {
      const stk::mesh::Selector masterA = periodicSelectorPairs_[0].first;
      const stk::mesh::Selector masterB = periodicSelectorPairs_[1].first;
      const stk::mesh::Selector masterC = periodicSelectorPairs_[2].first;

      const stk::mesh::Selector slaveA = periodicSelectorPairs_[0].second;
      const stk::mesh::Selector slaveB = periodicSelectorPairs_[1].second;
      const stk::mesh::Selector slaveC = periodicSelectorPairs_[2].second;

      // push back intersection selector pairs
      periodicSelectorPairs_.push_back(std::make_pair(masterA & masterB, slaveA & slaveB));
      periodicSelectorPairs_.push_back(std::make_pair(masterB & masterC, slaveB & slaveC));
      periodicSelectorPairs_.push_back(std::make_pair(masterA & masterC, slaveA & slaveC));
      periodicSelectorPairs_.push_back(std::make_pair(masterA & masterB & masterC, slaveA & slaveB & slaveC));

      // need a search method; arbitrarily choose the first method specified
      stk::search::SearchMethod searchMethod = searchMethodVec_[0];
      searchMethodVec_.push_back(searchMethod); // 3
      searchMethodVec_.push_back(searchMethod); // 4
      searchMethodVec_.push_back(searchMethod); // 5
      searchMethodVec_.push_back(searchMethod); // 6

      break;
    }

    default: {
      NaluEnv::self().naluOutputP0() << "more than three periodic pairs assumes no common slave nodes " << std::endl;
      break;
    }
  }
}

//--------------------------------------------------------------------------
//-------- initialize_translation_vector -----------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::initialize_translation_vector()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  translationVector_.resize(periodicSelectorPairs_.size());
  rotationVector_.resize(periodicSelectorPairs_.size());
  const int nDim = meta_data.spatial_dimension();
  for ( size_t k = 0; k < periodicSelectorPairs_.size(); ++k ) {
    translationVector_[k].resize(nDim);
    rotationVector_[k].resize(nDim);
  }
}

//--------------------------------------------------------------------------
//-------- determine_translation -------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::determine_translation(
    stk::mesh::Selector masterSelector,
    stk::mesh::Selector slaveSelector,
    std::vector<double> &translationVector,
    std::vector<double> &rotationVector)
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // fields
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  const int nDim = meta_data.spatial_dimension();

  // Master: global_sum_coords_master
  std::vector<double> local_sum_coords_master(nDim, 0.0), global_sum_coords_master(nDim, 0.0);
  size_t numberMasterNodes = 0, g_numberMasterNodes = 0;

  stk::mesh::BucketVector const& master_node_buckets = realm_.get_buckets( stk::topology::NODE_RANK, masterSelector);

  for ( stk::mesh::BucketVector::const_iterator ib = master_node_buckets.begin();
        ib != master_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    const stk::mesh::Bucket::size_type length   = b.size();
    // point to data
    const double * coords = stk::mesh::field_data(*coordinates, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      numberMasterNodes += 1;

      // define offset for all nodal fields that are of nDim
      const size_t offSet = k*nDim;

      // sum local coords for translation; define localCoords for bounding point
      for (int j = 0; j < nDim; ++j ) {
        const double cxj = coords[offSet+j];
        local_sum_coords_master[j] += cxj;
      }
    }
  }
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &local_sum_coords_master[0], &global_sum_coords_master[0], nDim);
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &numberMasterNodes, &g_numberMasterNodes, 1);

  // Slave: global_sum_coords_slave
  std::vector<double> local_sum_coords_slave(nDim, 0.0), global_sum_coords_slave(nDim, 0.0);
  size_t numberSlaveNodes = 0, g_numberSlaveNodes = 0;

  stk::mesh::BucketVector const& slave_node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, slaveSelector);

  for ( stk::mesh::BucketVector::const_iterator ib = slave_node_buckets.begin();
        ib != slave_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    const stk::mesh::Bucket::size_type length   = b.size();
    // point to data
    const double * coords = stk::mesh::field_data(*coordinates, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      numberSlaveNodes += 1;
      // define offset for all nodal fields that are of nDim
      const size_t offSet = k*nDim;
      for (int j = 0; j < nDim; ++j ) {
        local_sum_coords_slave[j] += coords[offSet+j];
      }
    }
  }
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &local_sum_coords_slave[0], &global_sum_coords_slave[0], nDim);
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &numberSlaveNodes, &g_numberSlaveNodes, 1);

  // save off translation and rotation
  for (int j = 0; j < nDim; ++j ) {
    translationVector[j] = (global_sum_coords_master[j] - global_sum_coords_slave[j]) / g_numberMasterNodes;
    rotationVector[j] = global_sum_coords_master[j] / g_numberMasterNodes;
  }

  NaluEnv::self().naluOutputP0() << "Translating [ ";
  for (int j = 0; j < nDim; ++j ) {  NaluEnv::self().naluOutputP0() << translationVector[j] << " "; }
  NaluEnv::self().naluOutputP0() << "] Master/Slave pair " << std::endl;
}

//--------------------------------------------------------------------------
//-------- remove_redundant_slave_nodes ------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::remove_redundant_slave_nodes()
{

  switch (periodicSelectorPairs_.size()) {

    case 1: case 2:
      break; // nothing to do

    case 3: {
      // slave selectors
      stk::mesh::Selector &slaveA = periodicSelectorPairs_[0].second;
      stk::mesh::Selector &slaveB = periodicSelectorPairs_[1].second;

      // intersection of A/B
      stk::mesh::Selector slaveIntersection = slaveA & slaveB;

      // now remove redundant [corner] nodes
      periodicSelectorPairs_[0].second = slaveA - slaveIntersection;
      periodicSelectorPairs_[1].second = slaveB - slaveIntersection;

      break;
    }

    case 7: {
      // slave selectors
      const stk::mesh::Selector slaveA = periodicSelectorPairs_[0].second;
      const stk::mesh::Selector slaveB = periodicSelectorPairs_[1].second;
      const stk::mesh::Selector slaveC = periodicSelectorPairs_[2].second;

      // intersection of A/B/C (corner nodes)
      const stk::mesh::Selector slaveABC = slaveA & slaveB & slaveC;

      // intersection of A/B/C (edges of box)
      const stk::mesh::Selector slaveAB = slaveA & slaveB;
      const stk::mesh::Selector slaveAC = slaveA & slaveC;
      const stk::mesh::Selector slaveBC = slaveB & slaveC;

      // now remove redundant [corner] nodes
      periodicSelectorPairs_[0].second = slaveA - (slaveAB | slaveAC);
      periodicSelectorPairs_[1].second = slaveB - (slaveAB | slaveBC);
      periodicSelectorPairs_[2].second = slaveC - (slaveAC | slaveBC);

      // now remove redundant [edges of box] nodes
      periodicSelectorPairs_[3].second = slaveAB - slaveABC;
      periodicSelectorPairs_[4].second = slaveBC - slaveABC;
      periodicSelectorPairs_[5].second = slaveAC - slaveABC;

      break;
    }
    default: {
      NaluEnv::self().naluOutputP0() << "more than three periodic pairs assumes no common slave nodes " << std::endl;
      break;
    }
  }
}

//--------------------------------------------------------------------------
//-------- finalize_search -------------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::finalize_search()
{
  // clear vectors
  searchKeyVector_.clear();
  masterSlaveCommunicator_.clear();

  // process each pair
  for ( size_t k = 0; k < periodicSelectorPairs_.size(); ++k) {
    populate_search_key_vec(periodicSelectorPairs_[k].first, periodicSelectorPairs_[k].second,
                            translationVector_[k], searchMethodVec_[k]);
  }
  
  // manage ghosting
  manage_ghosting_object();

  // check to see if we need to attempt once more
  error_check();
}

//--------------------------------------------------------------------------
//-------- populate_search_key_vec -----------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::populate_search_key_vec(
    stk::mesh::Selector masterSelector,
    stk::mesh::Selector slaveSelector,
    std::vector<double> &translationVector,
    const stk::search::SearchMethod searchMethod)
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  // required data structures; master/slave
  std::vector<sphereBoundingBox> sphereBoundingBoxMasterVec;
  std::vector<sphereBoundingBox> sphereBoundingBoxSlaveVec;

  // fields
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  const int nDim = meta_data.spatial_dimension();

  // Point
  const double pointRadius = searchTolerance_;
  Point masterCenter, slaveCenter;

  // Master: setup sphereBoundingBoxMasterVec,
  stk::mesh::BucketVector const& master_node_buckets = realm_.get_buckets( stk::topology::NODE_RANK, masterSelector);

  for ( stk::mesh::BucketVector::const_iterator ib = master_node_buckets.begin();
        ib != master_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    const stk::mesh::Bucket::size_type length   = b.size();
    // point to data
    const double * coords = stk::mesh::field_data(*coordinates, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity node = b[k];
      // setup ident
      theEntityKey theIdent(bulk_data.entity_key(node), NaluEnv::self().parallel_rank());

      // define offset for all nodal fields that are of nDim
      const size_t offSet = k*nDim;

      // sum local coords for translation; define localCoords for bounding point
      for (int j = 0; j < nDim; ++j ) {
        const double cxj = coords[offSet+j];
        masterCenter[j] = cxj;
      }

      // create the bounding point sphere and push back
      sphereBoundingBox theSphere( Sphere(masterCenter, pointRadius), theIdent);
      sphereBoundingBoxMasterVec.push_back(theSphere);
    }
  }

  // SLAVE: setup sphereBoundingBoxSlaveVec; translate slave onto master
  stk::mesh::BucketVector const& slave_node_buckets =
  realm_.get_buckets( stk::topology::NODE_RANK, slaveSelector);
  for ( stk::mesh::BucketVector::const_iterator ib = slave_node_buckets.begin();
       ib != slave_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    const stk::mesh::Bucket::size_type length   = b.size();
    // point to data
    const double * coords = stk::mesh::field_data(*coordinates, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity node = b[k];
      // setup ident
      theEntityKey theIdent(bulk_data.entity_key(node), NaluEnv::self().parallel_rank());
      // define offset for all nodal fields that are of nDim
      const size_t offSet = k*nDim;

      // create the bounding point that is translated, then push back
      for (int j = 0; j < nDim; ++j ) {
        const double xj = coords[offSet+j];
        slaveCenter[j] = xj + translationVector[j];
      }
      sphereBoundingBox theSphere(Sphere(slaveCenter, pointRadius), theIdent);
      sphereBoundingBoxSlaveVec.push_back(theSphere);
    }
  }

  // will want to stuff product of search to a single vector
  std::vector<std::pair<theEntityKey, theEntityKey> > searchKeyPair;
  double timeA = NaluEnv::self().nalu_time();
  stk::search::coarse_search(sphereBoundingBoxSlaveVec, sphereBoundingBoxMasterVec, searchMethod, NaluEnv::self().parallel_comm(), searchKeyPair);
  timerSearch_ += (NaluEnv::self().nalu_time() - timeA);

  // populate searchKeyVector_; culmination of all master/slaves
  searchKeyVector_.insert(searchKeyVector_.end(), searchKeyPair.begin(), searchKeyPair.end());
}

//--------------------------------------------------------------------------
//-------- error_check -----------------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::error_check()
{
  // number of slave nodes should equal the size of the searchKeyVector_
  size_t l_totalNumber[2] = {0,0};

  // extract total locally owned slave nodes
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::Selector s_locally_owned = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(slavePartVector_);
  stk::mesh::BucketVector const& slave_node_buckets = realm_.get_buckets( stk::topology::NODE_RANK, s_locally_owned);
  for ( stk::mesh::BucketVector::const_iterator ib = slave_node_buckets.begin();
	ib != slave_node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    const stk::mesh::Bucket::size_type length   = b.size();
    l_totalNumber[0] += length;
  }
  
  // extract locally owned slave nodes from the search
  for (size_t i=0, size=searchKeyVector_.size(); i<size; ++i) {
    if ( NaluEnv::self().parallel_rank() == searchKeyVector_[i].second.proc())
      l_totalNumber[1] += 1;
  }
  
  // parallel sum and check
  size_t g_totalNumber[2] = {0,0};
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), l_totalNumber, g_totalNumber, 2);

  // hard error check
  if ( g_totalNumber[0] != g_totalNumber[1]) {
    // increment and report
    errorCount_++;
    NaluEnv::self().naluOutputP0() << "_____________________________________" << std::endl;
    NaluEnv::self().naluOutputP0() << "Probable issue with Search on attempt: " << errorCount_ << std::endl;
    NaluEnv::self().naluOutputP0() << "the total number of slave nodes (" << g_totalNumber[0] << ")" << std::endl;
    NaluEnv::self().naluOutputP0() << "does not equal the product of the search (" << g_totalNumber[1] << ")" << std::endl;

    // check to see if we should try again..
    if ( errorCount_ == maxErrorCount_ ) {
      NaluEnv::self().naluOutputP0() << "ABORT: Too many attempts; please check your mesh" << std::endl;
      throw std::runtime_error("PeriodiocBC::Error: Too many attempts; please check your mesh");
    }
    // reduce or increase search tolerance based on number of slaves and total
    if ( g_totalNumber[0] > g_totalNumber[1] ) {
      searchTolerance_ *= amplificationFactor_; // slave > total; increase tolerance
      NaluEnv::self().naluOutputP0() << "The algorithm will increase the search tolerance: " << searchTolerance_ << std::endl;
    }
    else {
      searchTolerance_ /= amplificationFactor_; // slave < total; reduce tolerance
      NaluEnv::self().naluOutputP0() << "The algorithm will reduce the search tolerance: " << searchTolerance_ << std::endl;
    }

    // try again
    finalize_search();
  }
  else {
    NaluEnv::self().naluOutputP0() << "---------------------------------------------------" << std::endl;
    NaluEnv::self().naluOutputP0() << "Parallel consistency noted in master/slave pairings: "
                                   << g_totalNumber[0] << "/"<< g_totalNumber[1] << std::endl;
    NaluEnv::self().naluOutputP0() << "---------------------------------------------------" << std::endl;
    NaluEnv::self().naluOutputP0() << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- manage_ghosting_object ------------------------------------------
//--------------------------------------------------------------------------
void add_range_nodes_to_sharers_of_domain_nodes(stk::mesh::BulkData& bulk_data,
                                                  const PeriodicManager::SearchKeyVector& searchKeyVector,
                                                  std::vector<stk::mesh::EntityProc>& sendNodes)
{
    stk::CommSparse commSparse(bulk_data.parallel());

    auto packingLambda = [&]() {
        int theRank = NaluEnv::self().parallel_rank();
        std::vector<int> sharingProcs;
        for(size_t i=0; i<searchKeyVector.size(); ++i) {
            int domainProc = searchKeyVector[i].first.proc();
            int rangeProc = searchKeyVector[i].second.proc();

            stk::mesh::EntityKey domainKey = searchKeyVector[i].first.id();
            stk::mesh::Entity domainNode = bulk_data.get_entity(domainKey);

            if ((theRank == domainProc) && bulk_data.bucket(domainNode).shared()) {
                stk::mesh::EntityId rangeId = searchKeyVector[i].second.id().id();
                stk::CommBuffer& sbuf = commSparse.send_buffer(rangeProc);

                bulk_data.comm_shared_procs(domainKey, sharingProcs);
                if (theRank == rangeProc) {
                    stk::mesh::Entity rangeNode = bulk_data.get_entity(stk::topology::NODE_RANK, rangeId);
                    for(int p : sharingProcs) {
                        sendNodes.push_back(stk::mesh::EntityProc(rangeNode, p));
                    }
                }
                else {
                    for(int p : sharingProcs) {
                        if (p != theRank && p != rangeProc) {
                            sbuf.pack(rangeId);
                            sbuf.pack(p);
                        }
                    }
                }
            }
        }
    };

    stk::pack_and_communicate(commSparse, packingLambda);

    stk::unpack_communications(commSparse, [&](int p)
    {
        stk::CommBuffer& rbuf = commSparse.recv_buffer(p);
        stk::mesh::EntityId gid;
        rbuf.unpack(gid);
        int proc;
        rbuf.unpack(proc);
        sendNodes.push_back(stk::mesh::EntityProc(bulk_data.get_entity(stk::topology::NODE_RANK, gid), proc));
    });
}

void
PeriodicManager::manage_ghosting_object()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  unsigned theRank = NaluEnv::self().parallel_rank();

  std::vector<stk::mesh::EntityProc> sendNodes;
  std::vector<stk::mesh::EntityProc> sendElems;
  std::vector<int> sharingProcs;
  for (size_t i=0, size=searchKeyVector_.size(); i<size; ++i) {

    unsigned domainProc = searchKeyVector_[i].first.proc();
    unsigned rangeProc = searchKeyVector_[i].second.proc();

    if ((theRank != domainProc) && (theRank == rangeProc)) {
      stk::mesh::Entity rangeNode = bulk_data.get_entity(searchKeyVector_[i].second.id());
      sendNodes.push_back(stk::mesh::EntityProc(rangeNode, domainProc));

      if (realm_.hypreIsActive_) {
        auto* elems = bulk_data.begin_elements(rangeNode);
        int nelems = bulk_data.num_elements(rangeNode);
        for (int ie=0; ie<nelems; ie++)
          if (bulk_data.bucket(elems[ie]).owned())
            sendElems.push_back(stk::mesh::EntityProc(elems[ie], domainProc));
      }
    }
    else if ((theRank == domainProc) && (theRank != rangeProc)) {
      stk::mesh::Entity domainNode = bulk_data.get_entity(searchKeyVector_[i].first.id());
      sendNodes.push_back(stk::mesh::EntityProc(domainNode, rangeProc));

      if (realm_.hypreIsActive_) {
        auto* elems = bulk_data.begin_elements(domainNode);
        int nelems = bulk_data.num_elements(domainNode);
        for (int ie=0; ie<nelems; ie++)
          if (bulk_data.bucket(elems[ie]).owned())
            sendElems.push_back(stk::mesh::EntityProc(elems[ie], rangeProc));
      }
    }
  }

  add_range_nodes_to_sharers_of_domain_nodes(bulk_data, searchKeyVector_, sendNodes);

  size_t numNodes = sendNodes.size();
  size_t g_numNodes = 0;
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &numNodes, &g_numNodes, 1);
  if ( g_numNodes > 0) {
    // check if we need to ghost
    bulk_data.modification_begin();
    if ( periodicGhosting_ == NULL )
      periodicGhosting_ = &bulk_data.create_ghosting(ghostingName_);
    else
      bulk_data.destroy_ghosting(*periodicGhosting_);
    bulk_data.change_ghosting(*periodicGhosting_, sendNodes);
    if (realm_.hypreIsActive_)
      bulk_data.change_ghosting(*periodicGhosting_, sendElems);
    bulk_data.modification_end();

    populate_ghost_comm_procs(bulk_data, *periodicGhosting_, ghostCommProcs_);
  }

  // now populate master slave communicator
  for (size_t i=0, size=searchKeyVector_.size(); i<size; ++i) {
     stk::mesh::Entity domainNode = bulk_data.get_entity(searchKeyVector_[i].first.id());
     stk::mesh::Entity rangeNode = bulk_data.get_entity(searchKeyVector_[i].second.id());
     // unique master:slave communicator
     EntityPair theFirstPair = std::make_pair(rangeNode, domainNode);
     masterSlaveCommunicator_.push_back(theFirstPair);
  }
}

//--------------------------------------------------------------------------
//-------- get_ghosting_object ------------------------------------------
//--------------------------------------------------------------------------
stk::mesh::Ghosting *
PeriodicManager::get_ghosting_object()
{
  return periodicGhosting_;
}

//--------------------------------------------------------------------------
//-------- periodic_parallel_communicate_field -----------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::periodic_parallel_communicate_field(
  stk::mesh::FieldBase *theField)
{
  if ( NULL != periodicGhosting_ ) {
    std::vector< const stk::mesh::FieldBase *> fieldVec(1, theField);
    stk::mesh::communicate_field_data(*periodicGhosting_, fieldVec);
  }
}

//--------------------------------------------------------------------------
//-------- parallel_communicate_field --------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::parallel_communicate_field(
  stk::mesh::FieldBase *theField)
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  const unsigned pSize = bulk_data.parallel_size();
  if ( pSize > 1 ) {
    std::vector< const stk::mesh::FieldBase *> fieldVec(1, theField);
    stk::mesh::copy_owned_to_shared( bulk_data, fieldVec);
    stk::mesh::communicate_field_data(bulk_data.aura_ghosting(), fieldVec);
  }
}

//--------------------------------------------------------------------------
//-------- update global_id_field ------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::update_global_id_field()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  // no need to update periodically ghosted fields..

  // vector of masterEntity:slaveEntity pairs
  for ( size_t k = 0; k < masterSlaveCommunicator_.size(); ++k) {

    // extract master node and slave node
    EntityPair vecPair = masterSlaveCommunicator_[k];
    const stk::mesh::Entity masterNode = vecPair.first;
    const stk::mesh::Entity slaveNode = vecPair.second;

    // pointer to data
    const stk::mesh::EntityId masterGlobalId = bulk_data.identifier(masterNode );
    stk::mesh::EntityId *slaveGlobalId = stk::mesh::field_data(*realm_.naluGlobalId_, slaveNode);

    // set new value
    *slaveGlobalId = masterGlobalId;

  }

  // update all shared; aura and periodic
  parallel_communicate_field(realm_.naluGlobalId_);
}

//--------------------------------------------------------------------------
//-------- apply_constraints -----------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::apply_constraints(
  stk::mesh::FieldBase *theField,
  const unsigned &sizeOfField,
  const bool &bypassFieldCheck,
  const bool &addSlaves,
  const bool &setSlaves)
{

  // update periodically ghosted fields within add_ and set_
  if ( addSlaves )
    add_slave_to_master(theField, sizeOfField, bypassFieldCheck);
  if ( setSlaves )
    set_slave_to_master(theField, sizeOfField, bypassFieldCheck);

  // parallel communicate shared and aura-ed entities
  parallel_communicate_field(theField);

}


//--------------------------------------------------------------------------
//-------- apply_max_field -------------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::apply_max_field(
  stk::mesh::FieldBase *theField,
  const unsigned &sizeOfField)
{

  periodic_parallel_communicate_field(theField);

  for ( size_t k = 0; k < masterSlaveCommunicator_.size(); ++k) {
    // extract master node and slave node
    EntityPair vecPair = masterSlaveCommunicator_[k];
    const stk::mesh::Entity masterNode = vecPair.first;
    const stk::mesh::Entity slaveNode = vecPair.second;
    // pointer to data
    double *masterField = (double *)stk::mesh::field_data(*theField, masterNode);
    double *slaveField = (double *)stk::mesh::field_data(*theField, slaveNode);

    for ( unsigned j = 0; j < sizeOfField; ++j ) {
      const double maxValue = std::max(masterField[j],slaveField[j]);
      masterField[j] = maxValue; 
      slaveField[j] = maxValue;
    }
  }

  // parallel communicate shared and aura-ed entities
  parallel_communicate_field(theField);

}

//--------------------------------------------------------------------------
//-------- add_slave_to_master ---------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::add_slave_to_master(
  stk::mesh::FieldBase *theField,
  const unsigned &sizeOfField,
  const bool &bypassFieldCheck)
{
  
  periodic_parallel_communicate_field(theField);

  // iterate vector of masterEntity:slaveEntity pairs
  if ( bypassFieldCheck ) {
    // fields are expected to be defined on all master/slave nodes
    for ( size_t k = 0; k < masterSlaveCommunicator_.size(); ++k) {
      // extract master node and slave node
      EntityPair vecPair = masterSlaveCommunicator_[k];
      const stk::mesh::Entity masterNode = vecPair.first;
      const stk::mesh::Entity slaveNode = vecPair.second;
      // pointer to data
      double *masterField = (double *)stk::mesh::field_data(*theField, masterNode);
      const double *slaveField = (double *)stk::mesh::field_data(*theField, slaveNode);
      // add in contribution
      for ( unsigned j = 0; j < sizeOfField; ++j ) {
        masterField[j] += slaveField[j];
      }
    }
  }
  else {
    // more costly check to see if fields are defined on master/slave nodes    
    for ( size_t k = 0; k < masterSlaveCommunicator_.size(); ++k) {      
      // extract master node and slave node
      EntityPair vecPair = masterSlaveCommunicator_[k];
      const stk::mesh::Entity masterNode = vecPair.first;
      const stk::mesh::Entity slaveNode = vecPair.second;
      // pointer to data
      double *masterField = (double *)stk::mesh::field_data(*theField, masterNode);
      if ( NULL != masterField ) {
        const double *slaveField = (double *)stk::mesh::field_data(*theField, slaveNode);
        // add in contribution
        for ( unsigned j = 0; j < sizeOfField; ++j ) {
          masterField[j] += slaveField[j];
        }
      }
    }
  }

  periodic_parallel_communicate_field(theField);

}

//--------------------------------------------------------------------------
//-------- set_slave_to_master ---------------------------------------------
//--------------------------------------------------------------------------
void
PeriodicManager::set_slave_to_master(
  stk::mesh::FieldBase *theField,
  const unsigned &sizeOfField,
  const bool &bypassFieldCheck)
{

  periodic_parallel_communicate_field(theField);

  // iterate vector of masterEntity:slaveEntity pairs
  if ( bypassFieldCheck ) {
    // fields are expected to be defined on all master/slave nodes
    for ( size_t k = 0; k < masterSlaveCommunicator_.size(); ++k) {
      // extract master node and slave node
      EntityPair vecPair = masterSlaveCommunicator_[k];
      const stk::mesh::Entity masterNode = vecPair.first;
      const stk::mesh::Entity slaveNode = vecPair.second;
      // pointer to data
      const double *masterField = (double *)stk::mesh::field_data(*theField, masterNode);
      double *slaveField = (double *)stk::mesh::field_data(*theField, slaveNode);
      // set master to slave
      for ( unsigned j = 0; j < sizeOfField; ++j ) {
        slaveField[j] = masterField[j];
      }
    }
  }
  else {
    // more costly check to see if fields are defined on master/slave nodes    
    for ( size_t k = 0; k < masterSlaveCommunicator_.size(); ++k) {
      // extract master node and slave node
      EntityPair vecPair = masterSlaveCommunicator_[k];
      const stk::mesh::Entity masterNode = vecPair.first;
      const stk::mesh::Entity slaveNode = vecPair.second;
      // pointer to data
      const double *masterField = (double *)stk::mesh::field_data(*theField, masterNode);
      
      if ( NULL != masterField ) {
        double *slaveField = (double *)stk::mesh::field_data(*theField, slaveNode);
        // set master to slave
        for ( unsigned j = 0; j < sizeOfField; ++j ) {
          slaveField[j] = masterField[j];
        }
      }
    }
  }

  periodic_parallel_communicate_field(theField);

}

} // namespace nalu
} // namespace sierra
