/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifdef NALU_USES_TIOGA

#include "overset/TiogaSTKIface.h"
#include "overset/TiogaBlock.h"

#include "overset/OversetManagerTIOGA.h"
#include "overset/OversetInfo.h"
#include <utils/StkHelpers.h>

#include "NaluEnv.h"
#include "Realm.h"
#include "master_element/MasterElement.h"
#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/SkinBoundary.hpp"


#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "tioga.h"

namespace tioga_nalu {

TiogaSTKIface::TiogaSTKIface(
  sierra::nalu::OversetManagerTIOGA& oversetManager,
  const YAML::Node& node
) : oversetManager_(oversetManager),
    meta_(*oversetManager.metaData_),
    bulk_(*oversetManager.bulkData_),
    tg_(new TIOGA::tioga()),
    inactivePartName_("nalu_overset_hole_elements")
{
  load(node);
}

TiogaSTKIface::~TiogaSTKIface()
{}

void
TiogaSTKIface::load(const YAML::Node& node)
{
  const YAML::Node& oset_groups = node["mesh_group"];

  std::string coords_name = oversetManager_.realm_.get_coordinates_name();
  int num_meshes = oset_groups.size();
  blocks_.resize(num_meshes);

  for (int i = 0; i < num_meshes; i++) {
    blocks_[i].reset(new TiogaBlock(meta_, bulk_, oset_groups[i], coords_name, i + 1));
  }

  sierra::nalu::NaluEnv::self().naluOutputP0()
      << "TIOGA: Using coordinates field: " << coords_name << std::endl;

  if (node["tioga_populate_inactive_part"])
    populateInactivePart_ = node["tioga_populate_inactive_part"].as<bool>();

  if (node["tioga_symmetry_direction"])
    symmetryDir_ = node["tioga_symmetry_direction"].as<int>();
}

void TiogaSTKIface::setup(stk::mesh::PartVector& bcPartVec)
{
  for (auto& tb: blocks_) {
    tb->setup(bcPartVec);
  }

  if (populateInactivePart_) {
    // Initialize the inactive part
    oversetManager_.inActivePart_ = &meta_.declare_part(
      inactivePartName_, stk::topology::ELEM_RANK);

    oversetManager_.backgroundSurfacePart_ = &meta_.declare_part(
      "nalu_overset_fringe_boundary", stk::topology::ELEM_RANK);
  }
}

void TiogaSTKIface::initialize()
{
  tg_->setCommunicator(bulk_.parallel(),
                       bulk_.parallel_rank(),
                       bulk_.parallel_size());

  tg_->setSymmetry(symmetryDir_);

  sierra::nalu::NaluEnv::self().naluOutputP0()
    << "TIOGA: Initializing overset mesh blocks: " << std::endl;
  for (auto& tb: blocks_) {
    tb->initialize();
  }
  sierra::nalu::NaluEnv::self().naluOutputP0()
    << "TIOGA: Initialized " << blocks_.size() << " overset blocks" << std::endl;
}

void TiogaSTKIface::initialize_ghosting()
{
  // TODO: Update ghosting modification to use optimized version in
  // Non-conformal case.
  stk::mesh::Ghosting* ovsetGhosting = oversetManager_.oversetGhosting_;
  bulk_.modification_begin();
  if (ovsetGhosting == nullptr) {
    const std::string ghostName = "nalu_overset_ghosting";
    oversetManager_.oversetGhosting_ = &(bulk_.create_ghosting(ghostName));
  } else {
    bulk_.destroy_ghosting(*(oversetManager_.oversetGhosting_));
  }
  bulk_.modification_end();
}

void TiogaSTKIface::execute()
{
  reset_data_structures();

  // Update the coordinates for TIOGA and register updates to the TIOGA mesh block.
  for (auto& tb: blocks_) {
    tb->update_coords();
    tb->register_block(*tg_);
  }

  // Determine overset connectivity
  tg_->profile();
  tg_->performConnectivity();

  for (auto& tb: blocks_) {
    // Update IBLANK information at nodes and elements
    tb->update_iblanks(oversetManager_.holeNodes_);
    tb->update_iblank_cell();

    // For each block determine donor elements that needs to be ghosted to other
    // MPI ranks
    tb->get_donor_info(*tg_, elemsToGhost_);
  }

  // Synchronize IBLANK data for shared nodes
  ScalarIntFieldType* ibf = meta_.get_field<ScalarIntFieldType>(
          stk::topology::NODE_RANK, "iblank");
  std::vector<const stk::mesh::FieldBase*> pvec{ibf};
  stk::mesh::copy_owned_to_shared(bulk_, pvec);

  get_receptor_info();

  // Collect all elements to be ghosted and update ghosting so that the elements
  // are available when generating {fringeNode, donorElement} pairs in the next
  // step.
  update_ghosting();

  if (populateInactivePart_) populate_inactive_part();

  // Update overset fringe connectivity information for Constraint based algorithm
  populate_overset_info();
}

void TiogaSTKIface::reset_data_structures()
{
  // Reset inactivePart_
  if (populateInactivePart_) {
    bulk_.modification_begin();
    stk::mesh::Part* inactivePart = oversetManager_.inActivePart_;
    stk::mesh::PartVector add_parts;
    stk::mesh::PartVector remove_parts;
    remove_parts.push_back(inactivePart);
    for (auto elem : holeElems_) {
      bulk_.change_entity_parts(elem, add_parts, remove_parts);
    }

    stk::mesh::PartVector fringe_remove_parts;
    fringe_remove_parts.push_back(oversetManager_.backgroundSurfacePart_);
    for (auto elem: fringeElems_) {
      bulk_.change_entity_parts(elem, add_parts, fringe_remove_parts);
    }
    bulk_.modification_end();
  }

  holeElems_.clear();
  fringeElems_.clear();
  elemsToGhost_.clear();
  donorIDs_.clear();
  receptorIDs_.clear();
  oversetManager_.orphanPointSurfaceVecBackground_.clear();
}

void TiogaSTKIface::update_ghosting()
{
  stk::mesh::Ghosting* ovsetGhosting = oversetManager_.oversetGhosting_;
  std::vector<stk::mesh::EntityKey> recvGhostsToRemove;

  if (ovsetGhosting != nullptr) {
    stk::mesh::EntityProcVec currentSendGhosts;
    ovsetGhosting->send_list(currentSendGhosts);

    sierra::nalu::compute_precise_ghosting_lists(
      bulk_, elemsToGhost_, currentSendGhosts, recvGhostsToRemove);
  }

  size_t local[2] = {elemsToGhost_.size(), recvGhostsToRemove.size()};
  size_t global[2] = {0, 0};
  stk::all_reduce_sum(bulk_.parallel(), local, global, 2);

  if ((global[0] > 0) || (global[1] > 0)) {
    bulk_.modification_begin();
    if (ovsetGhosting == nullptr) {
      const std::string ghostName = "nalu_overset_ghosting";
      oversetManager_.oversetGhosting_ = &(bulk_.create_ghosting(ghostName));
    }
    bulk_.change_ghosting(
      *(oversetManager_.oversetGhosting_), elemsToGhost_, recvGhostsToRemove);
    bulk_.modification_end();

    sierra::nalu::populate_ghost_comm_procs(
      bulk_, *(oversetManager_.oversetGhosting_), oversetManager_.ghostCommProcs_);

#if 1
    sierra::nalu::NaluEnv::self().naluOutputP0()
      << "TIOGA: Overset algorithm will ghost " << global[0] << " elements"
      << std::endl;
#endif
  }
#if 1
  else {
    sierra::nalu::NaluEnv::self().naluOutputP0()
      << "TIOGA: Overset ghosting unchanged for this timestep" << std::endl;
  }
#endif
}

void TiogaSTKIface::populate_inactive_part()
{
  stk::mesh::PartVector toParts;
  stk::mesh::PartVector fringeParts;
  toParts.push_back(oversetManager_.inActivePart_);
  fringeParts.push_back(oversetManager_.backgroundSurfacePart_);

  // Gather all the "hole" elements that are not solved for, or cannot have a
  // valid solution because the nodes lie within a solid body and add it to the
  // inactive part. The rows in the linear system matrix for these nodes will be
  // removed during the linear system reinitialization.
  bulk_.modification_begin();
  {
    for (auto& tb: blocks_) {
      auto iblank_cell = tb->iblank_cell();
      auto elem_gid = tb->elem_id_map();

      for (size_t i=0; i<iblank_cell.size(); i++) {
        auto ib = iblank_cell[i];

        if (ib == 0) {
          stk::mesh::Entity elem = bulk_.get_entity(
            stk::topology::ELEM_RANK, elem_gid[i]);
          bulk_.change_entity_parts(elem, toParts);
          holeElems_.push_back(elem);
        }
        else if (ib == -1) {
          stk::mesh::Entity elem = bulk_.get_entity(
            stk::topology::ELEM_RANK, elem_gid[i]);
          bulk_.change_entity_parts(elem, fringeParts);
          fringeElems_.push_back(elem);
        }
      }
    }
  }
  bulk_.modification_end();

  oversetManager_.orphanPointSurfaceVecBackground_.push_back(
    oversetManager_.backgroundSurfacePart_);

#if 1
  size_t numHolesLocal = holeElems_.size();
  size_t numHolesGlobal = 0;
  stk::all_reduce_sum(bulk_.parallel(), &numHolesLocal, 
          &numHolesGlobal, 1);
  sierra::nalu::NaluEnv::self().naluOutputP0()
      << "TIOGA: Num. inactive elements: " << numHolesGlobal << std::endl;
#endif
}

void TiogaSTKIface::update_fringe_info()
{
  auto& realm = oversetManager_.realm_;
  auto& osetInfo = oversetManager_.oversetInfoVec_;
  int nDim = meta_.spatial_dimension();
  std::vector<double> elemCoords;

  // Ask TIOGA for the fringe points and their corresponding donor element
  // information
  std::vector<int> receptors;
  tg_->getReceptorInfo(receptors);

  // Before we begin ensure that we have cleaned up oversetInfoVec
  ThrowAssert(osetInfo.size() == 0);

  VectorFieldType *coords = meta_.get_field<VectorFieldType>
    (stk::topology::NODE_RANK, realm.get_coordinates_name());
  ScalarIntFieldType* ibf = meta_.get_field<ScalarIntFieldType>(
          stk::topology::NODE_RANK, "iblank");
  int nbadnodes = 0;

  // Process TIOGA receptors array and fill in the oversetInfoVec used for
  // subsequent Nalu computations.
  //
  // TIOGA returns a integer array that contains 3 entries per receptor node:
  //   - the local node index within the tioga mesh data array
  //   - the local mesh tag (block index) for that mesh during registration
  //   - the STK global ID for the donor element
  //
  size_t ncount = receptors.size();
  for (size_t i=0; i<ncount; i+=3) {
    int nid = receptors[i];                          // TiogaBlock node index
    int mtag = receptors[i+1] - 1;                   // Block index
    int donorID = receptors[i+2];                    // STK Global ID of the donor element
    int nodeID = blocks_[mtag]->node_id_map()[nid];  // STK Global ID of the fringe node
    stk::mesh::Entity node = bulk_.get_entity(stk::topology::NODE_RANK, nodeID);
    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, donorID);

    if (!bulk_.bucket(node).owned()) {
        int ibval = *stk::mesh::field_data(*ibf, node);

        if (ibval > -1) {
            nbadnodes++;
            std::cerr << mtag << "\t" << nodeID << "\t" << donorID 
                << "\t" << ibval << std::endl;
        }
        //continue;
    }

#ifndef NDEBUG
    // The donor element must have already been ghosted to the required MPI rank,
    // so validity check should always succeed.
    if (!bulk_.is_valid(elem))
      throw std::runtime_error(
        "Invalid element encountered in overset mesh connectivity");
#endif

    // At this point, we have all the necessary information to create an
    // OversetInfo instance for this {receptor node, donor element} pair.
    sierra::nalu::OversetInfo* oinfo = new sierra::nalu::OversetInfo(node, nDim);
    osetInfo.push_back(oinfo);

    // Store away the coordinates for this receptor node for later use
    const double* xyz = stk::mesh::field_data(*coords, node);
    for (int i=0; i<nDim; i++) {
      oinfo->nodalCoords_[i] = xyz[i];
    }

    const stk::topology elemTopo = bulk_.bucket(elem).topology();
    const stk::mesh::Entity* enodes = bulk_.begin_nodes(elem);
    sierra::nalu::MasterElement* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
    int num_nodes = bulk_.num_nodes(elem);
    elemCoords.resize(nDim*num_nodes);

    for (int ni=0; ni < num_nodes; ++ni) {
      stk::mesh::Entity enode = enodes[ni];
      const double* xyz = stk::mesh::field_data(*coords, enode);
      for (int j=0; j < nDim; j++) {
        const int offset = j * num_nodes + ni;
        elemCoords[offset] = xyz[j];
      }
    }

    const double nearestDistance = meSCS->isInElement(
      elemCoords.data(),
      oinfo->nodalCoords_.data(),
      oinfo->isoParCoords_.data());

#if 0
    if (nearestDistance > (1.0 + 1.0e-8))
      sierra::nalu::NaluEnv::self().naluOutput()
        << "TIOGA WARNING: In pair (" << nodeID << ", " << donorID << "): "
        << "iso-parametric distance is greater than 1.0: " << nearestDistance
        << std::endl;
#endif

    oinfo->owningElement_ = elem;
    oinfo->meSCS_ = meSCS;
    oinfo->bestX_ = nearestDistance;
    oinfo->elemIsGhosted_ = bulk_.bucket(elem).owned()? 0 : 1;
  }

#if 1
  // Debugging information
  size_t numFringeLocal = osetInfo.size();
  size_t numFringeGlobal = 0;
  stk::all_reduce_sum(bulk_.parallel(), &numFringeLocal, &numFringeGlobal, 1);

  sierra::nalu::NaluEnv::self().naluOutputP0()
    << "TIOGA: Num. receptor nodes = " << numFringeGlobal
    << std::endl;

  int nbadnodesGlobal = 0;
  stk::all_reduce_sum(bulk_.parallel(), &nbadnodes, &nbadnodesGlobal, 1);
  sierra::nalu::NaluEnv::self().naluOutputP0()
    << "TIOGA: Num. inconsistent nodes = " << nbadnodesGlobal
    << std::endl;

#endif
}

void
TiogaSTKIface::get_receptor_info()
{
  ScalarIntFieldType* ibf = meta_.get_field<ScalarIntFieldType>(
    stk::topology::NODE_RANK, "iblank");

  std::vector<unsigned long> nodesToReset;

  // Ask TIOGA for the fringe points and their corresponding donor element
  // information
  std::vector<int> receptors;
  tg_->getReceptorInfo(receptors);

  // Process TIOGA receptors array and fill in the oversetInfoVec used for
  // subsequent Nalu computations.
  //
  // TIOGA returns a integer array that contains 3 entries per receptor node:
  //   - the local node index within the tioga mesh data array
  //   - the local mesh tag (block index) for that mesh during registration
  //   - the STK global ID for the donor element (can be 8-byte or 4-byte)
  //
  size_t ncount = receptors.size();
  stk::mesh::EntityId  donorID = std::numeric_limits<stk::mesh::EntityId>::max();
#ifdef TIOGA_HAS_UINT64T
  // The donor ID is stored in 2 4-byte integer entries (2 + 2 = 4). See above
  // for description on what is returned for each receptor node.
  const int rec_offset = 4;
#else
  // The donor ID is stored in a single 4-byte integer entry (2 + 1 = 3)
  const int rec_offset = 3;
#endif
  for (size_t i=0; i<ncount; i+=rec_offset) {
    int nid = receptors[i];                          // TiogaBlock node index
    int mtag = receptors[i+1] - 1;                   // Block index
#ifdef TIOGA_HAS_UINT64T
    std::memcpy(&donorID, &receptors[i+2], sizeof(uint64_t));
#else
    donorID = receptors[i+2];                    // STK Global ID of the donor element
#endif
    auto nodeID = blocks_[mtag]->node_id_map()[nid]; // STK Global ID of the fringe node
    stk::mesh::Entity node = bulk_.get_entity(stk::topology::NODE_RANK, nodeID);

    if (!bulk_.bucket(node).owned()) {
      // We have a shared node that is marked as fringe. Ensure that the owning
      // proc also has this marked as fringe.
      int ibval = *stk::mesh::field_data(*ibf, node);

      if (ibval > -1) {
        // Disagreement between owner and shared status of iblank. Communicate
        // to owner and other shared procs that it must be a fringe.
        std::vector<int> sprocs;
        bulk_.comm_shared_procs(bulk_.entity_key(node), sprocs);
        for (auto jproc: sprocs) {
          if (jproc == bulk_.parallel_rank()) continue;

          nodesToReset.push_back(jproc);
          nodesToReset.push_back(nodeID);
          nodesToReset.push_back(donorID);
        }
      }
    }

    // Stash the IDs for populating OversetInfo
    donorIDs_.push_back(donorID);
    receptorIDs_.push_back(nodeID);
  }

  int numLocal = nodesToReset.size();
  int iproc = bulk_.parallel_rank();
  int nproc = bulk_.parallel_size();
  std::vector<int> nbPerProc(nproc);
  MPI_Allgather(&numLocal, 1, MPI_INT, nbPerProc.data(), 1, MPI_INT, bulk_.parallel());

  // Total number of entities across all procs
  int nTotalEntities = std::accumulate(nbPerProc.begin(), nbPerProc.end(), 0);

  // If no disagreements were detected then we are done here
  if (nTotalEntities < 1) return;

#if 1
  sierra::nalu::NaluEnv::self().naluOutputP0()
    << "TIOGA: Detected fringe/field mismatch on " << (nTotalEntities/3)
    << " entities" << std::endl;
#endif

  // Prepare data structures for reconciliation
  std::vector<int> offsets(nproc+1);
  std::vector<unsigned long> allEntities(nTotalEntities);

  offsets[0] = 0;
  for (int i=1; i <= nproc; ++i) {
    offsets[i] = offsets[i-1] + nbPerProc[i-1];
  }

  MPI_Allgatherv(nodesToReset.data(), numLocal, MPI_UNSIGNED_LONG, allEntities.data(),
                 nbPerProc.data(), offsets.data(), MPI_UNSIGNED_LONG, bulk_.parallel());

  for (int i=0; i < nTotalEntities; i+=3) {
    int nodeProc = allEntities[i];
    stk::mesh::EntityId nodeID = allEntities[i+1];
    stk::mesh::EntityId donorID = allEntities[i+2];

    // Add the receptor donor pair to populate OversetInfo
    if (iproc == nodeProc) {
      receptorIDs_.push_back(nodeID);
      donorIDs_.push_back(donorID);
    }

    // Setup for ghosting
    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, donorID);
    if (bulk_.is_valid(elem) &&
        (bulk_.parallel_owner_rank(elem) == iproc) &&
        (nodeProc != iproc)) {
      // Found the owning proc for this donor element. Request ghosting
      stk::mesh::EntityProc elem_proc(elem, nodeProc);
      elemsToGhost_.push_back(elem_proc);
    }
  }
}

void
TiogaSTKIface::populate_overset_info()
{
  auto& realm = oversetManager_.realm_;
  auto& osetInfo = oversetManager_.oversetInfoVec_;
  int nDim = meta_.spatial_dimension();
  std::vector<double> elemCoords;
  std::unordered_set<stk::mesh::EntityId> seenIDs;

  // Ensure that the oversetInfoVec has been cleared out
  ThrowAssert(osetInfo.size() == 0);

  VectorFieldType *coords = meta_.get_field<VectorFieldType>
    (stk::topology::NODE_RANK, realm.get_coordinates_name());

  size_t numReceptors = receptorIDs_.size();
  for (size_t i=0; i < numReceptors; i++) {
    stk::mesh::EntityId nodeID = receptorIDs_[i];
    stk::mesh::EntityId donorID = donorIDs_[i];
    stk::mesh::Entity node = bulk_.get_entity(stk::topology::NODE_RANK, nodeID);
    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, donorID);

    // Track fringe nodes that have already been processed.
    //
    // This is necessary when handling fringe-field mismatch across processors,
    // multiple shared procs might indicate that the owner must reset their
    // status. This check ensures the fringe is processed only once.
    auto hasIt = seenIDs.find(nodeID);
    if (hasIt != seenIDs.end()) continue;
    seenIDs.insert(nodeID);

#if 1
    // The donor element must have already been ghosted to the required MPI
    // rank, so validity check should always succeed.
    if (!bulk_.is_valid(elem))
      throw std::runtime_error(
        "Invalid element encountered in overset mesh connectivity");
#endif

    // At this point, we have all the necessary information to create an
    // OversetInfo instance for this {receptor node, donor element} pair.
    sierra::nalu::OversetInfo* oinfo = new sierra::nalu::OversetInfo(node, nDim);
    osetInfo.push_back(oinfo);

    // Store away the coordinates for this receptor node for later use
    const double* xyz = stk::mesh::field_data(*coords, node);
    for (int i=0; i<nDim; i++) {
      oinfo->nodalCoords_[i] = xyz[i];
    }

    const stk::topology elemTopo = bulk_.bucket(elem).topology();
    const stk::mesh::Entity* enodes = bulk_.begin_nodes(elem);
    sierra::nalu::MasterElement* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
    int num_nodes = bulk_.num_nodes(elem);
    elemCoords.resize(nDim*num_nodes);

    for (int ni=0; ni < num_nodes; ++ni) {
      stk::mesh::Entity enode = enodes[ni];
      const double* xyz = stk::mesh::field_data(*coords, enode);
      for (int j=0; j < nDim; j++) {
        const int offset = j * num_nodes + ni;
        elemCoords[offset] = xyz[j];
      }
    }

    const double nearestDistance = meSCS->isInElement(
      elemCoords.data(),
      oinfo->nodalCoords_.data(),
      oinfo->isoParCoords_.data());

#if 0
    if (nearestDistance > (1.0 + 1.0e-8))
      sierra::nalu::NaluEnv::self().naluOutput()
        << "TIOGA WARNING: In pair (" << nodeID << ", " << donorID << "): "
        << "iso-parametric distance is greater than 1.0: " << nearestDistance
        << std::endl;
#endif

    oinfo->owningElement_ = elem;
    oinfo->meSCS_ = meSCS;
    oinfo->bestX_ = nearestDistance;
    oinfo->elemIsGhosted_ = bulk_.bucket(elem).owned()? 0 : 1;
  }

#if 1
  // Debugging information
  size_t numFringeLocal = osetInfo.size();
  size_t numFringeGlobal = 0;
  stk::all_reduce_sum(bulk_.parallel(), &numFringeLocal, &numFringeGlobal, 1);

  sierra::nalu::NaluEnv::self().naluOutputP0()
    << "TIOGA: Num. receptor nodes = " << numFringeGlobal
    << std::endl;
#endif
}

} // tioga

#endif // NALU_USES_TIOGA
