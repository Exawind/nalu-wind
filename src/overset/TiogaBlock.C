// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef NALU_USES_TIOGA

#include "overset/TiogaBlock.h"
#include "overset/OversetNGP.h"
#include "NaluEnv.h"

#include <stk_util/parallel/ParallelReduce.hpp>

#include "tioga.h"

#include <numeric>
#include <iostream>
#include <limits>
#include <algorithm>

namespace tioga_nalu {

TiogaBlock::TiogaBlock(
  stk::mesh::MetaData& meta,
  stk::mesh::BulkData& bulk,
  TiogaOptions& opts,
  const YAML::Node& node,
  const std::string coords_name,
  const int meshtag)
  : meta_(meta),
    bulk_(bulk),
    tiogaOpts_(opts),
    coords_name_(coords_name),
    ndim_(meta_.spatial_dimension()),
    meshtag_(meshtag),
    block_name_("tioga_block_" + std::to_string(meshtag_))
{
  load(node);
}

TiogaBlock::~TiogaBlock()
{
  if (tioga_conn_ != nullptr) {
    delete[] tioga_conn_;
  }
}

void
TiogaBlock::load(const YAML::Node& node)
{
  // Every participating mesh must register the mesh part
  blkNames_ = node["mesh_parts"].as<std::vector<std::string>>();

  // Wall and overset side-sets are optional for each participating block

  if (node["wall_parts"]) {
    wallNames_ = node["wall_parts"].as<std::vector<std::string>>();
  }

  if (node["ovset_parts"]) {
    ovsetNames_ = node["ovset_parts"].as<std::vector<std::string>>();
  }

  // Parse block name for informational messages.
  if (node["overset_name"]) {
    block_name_ = node["overset_name"].as<std::string>();
  }
}

void
TiogaBlock::setup(stk::mesh::PartVector& bcPartVec)
{
  names_to_parts(blkNames_, blkParts_);

  if (wallNames_.size() > 0)
    names_to_parts(wallNames_, wallParts_);

  if (ovsetNames_.size() > 0)
    names_to_parts(ovsetNames_, ovsetParts_);

  sierra::nalu::ScalarIntFieldType& ibf =
    meta_.declare_field<int>(stk::topology::NODE_RANK, "iblank");

  sierra::nalu::ScalarIntFieldType& ibcell =
    meta_.declare_field<int>(stk::topology::ELEM_RANK, "iblank_cell");

  sierra::nalu::ScalarFieldType& nVol =
    meta_.declare_field<double>(stk::topology::NODE_RANK, "tioga_nodal_volume");

  for (auto p : blkParts_) {
    stk::mesh::put_field_on_mesh(ibf, *p, nullptr);
    stk::mesh::put_field_on_mesh(ibcell, *p, nullptr);
    stk::mesh::put_field_on_mesh(nVol, *p, nullptr);
  }

  // Push overset BC parts to the realm_.bcPartVec_ so that they are ignored
  // when checking for missing BCs
  if (ovsetNames_.size() > 0)
    for (auto bcPart : ovsetParts_)
      bcPartVec.push_back(bcPart);
}

void
TiogaBlock::initialize()
{
  process_nodes();
  process_wallbc();
  process_ovsetbc();
  process_elements();

  print_summary();

  is_init_ = false;
}

void
TiogaBlock::update_coords()
{
  stk::mesh::Selector mesh_selector = get_node_selector(blkParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);
  sierra::nalu::VectorFieldType* coords =
    meta_.get_field<double>(stk::topology::NODE_RANK, coords_name_);
  sierra::nalu::ScalarFieldType* nodeVol =
    meta_.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");

#if 0
  std::vector<double> bboxMin(3);
  std::vector<double> bboxMax(3);

  for (int i=0; i<ndim_; i++) {
      bboxMin[i] = std::numeric_limits<double>::max();
      bboxMax[i] = -std::numeric_limits<double>::max();
  }
#endif

  auto& ngp_xyz = bdata_.xyz_.h_view;
  auto& noderes = bdata_.node_res_.h_view;
  const double fac = tiogaOpts_.node_res_mult();
  int ip = 0;
  for (auto b : mbkts) {
    for (size_t in = 0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];

      double* pt = stk::mesh::field_data(*coords, node);
      for (int i = 0; i < ndim_; i++) {
        ngp_xyz(ip * ndim_ + i) = pt[i];

#if 0
        bboxMin[i] = std::min(pt[i], bboxMin[i]);
        bboxMax[i] = std::max(pt[i], bboxMax[i]);
#endif
      }

      double* nVol = stk::mesh::field_data(*nodeVol, node);
      noderes(ip) = *nVol * fac;
      ip++;
    }
  }

  bdata_.xyz_.sync_device();
  bdata_.node_res_.sync_device();

#if 0
  std::vector<double> gMin(3,0.0);
  std::vector<double> gMax(3,0.0);
  stk::all_reduce_min(bulk_.parallel(), bboxMin.data(), gMin.data(), 3);
  stk::all_reduce_max(bulk_.parallel(), bboxMax.data(), gMax.data(), 3);

  sierra::nalu::NaluEnv::self().naluOutputP0()
      << "TIOGA: " << block_name_ << ": \n"
      << "\t" << gMin[0] << ", " << gMin[1] << ", " << gMin[2] << "\n"
      << "\t" << gMax[0] << ", " << gMax[1] << ", " << gMax[2] 
      << std::endl;
#endif
}

void
TiogaBlock::update_element_volumes()
{
  stk::mesh::Selector mesh_selector = get_elem_selector(blkParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::ELEM_RANK, mesh_selector);
  sierra::nalu::ScalarFieldType* elemVolume =
    meta_.get_field<double>(stk::topology::ELEMENT_RANK, "element_volume");

  auto& numverts = bdata_.num_verts_.h_view;
  auto& numcells = bdata_.num_cells_.h_view;
  const int ntypes = conn_map_.size();
  std::map<int, size_t> elem_offsets;
  size_t eoffset = 0;
  for (int i = 0; i < ntypes; i++) {
    int idx = numverts(i);
    elem_offsets[idx] = eoffset;
    eoffset += numcells(i);
  }

  auto& cellres = bdata_.cell_res_.h_view;
  const double fac = tiogaOpts_.cell_res_mult();
  for (auto b : mbkts) {
    double* eVol = stk::mesh::field_data(*elemVolume, *b);
    const int npe = b->topology().num_nodes();
    int ep = elem_offsets[npe];

    for (size_t ie = 0; ie < b->size(); ++ie)
      cellres(ep++) = eVol[ie] * fac;

    elem_offsets[npe] = ep;
  }
  bdata_.cell_res_.sync_device();
}

void
TiogaBlock::update_connectivity()
{
  process_nodes();
  process_wallbc();
  process_ovsetbc();
  process_elements();
}

void
TiogaBlock::update_iblanks(
  std::vector<stk::mesh::Entity>& holeNodes,
  std::vector<stk::mesh::Entity>& fringeNodes)
{
  sierra::nalu::ScalarIntFieldType* ibf =
    meta_.get_field<int>(stk::topology::NODE_RANK, "iblank");

  stk::mesh::Selector mesh_selector = get_node_selector(blkParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);

  auto& ibnode = bdata_.iblank_.h_view;
  int ip = 0;
  for (auto b : mbkts) {
    int* ib = stk::mesh::field_data(*ibf, *b);
    for (size_t in = 0; in < b->size(); in++) {
      ib[in] = ibnode(ip++);

      if (ib[in] == 0) {
        holeNodes.push_back((*b)[in]);
      } else if (ib[in] == -1) {
        fringeNodes.push_back((*b)[in]);
      }
    }
  }
}

void
TiogaBlock::update_iblank_cell()
{
  sierra::nalu::ScalarIntFieldType* ibf =
    meta_.get_field<int>(stk::topology::ELEM_RANK, "iblank_cell");

  stk::mesh::Selector mesh_selector = get_elem_selector(blkParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::ELEM_RANK, mesh_selector);

  auto& ibcell = bdata_.iblank_cell_.h_view;
  int ip = 0;
  for (auto b : mbkts) {
    int* ib = stk::mesh::field_data(*ibf, *b);
    for (size_t in = 0; in < b->size(); in++) {
      ib[in] = ibcell(ip++);
    }
  }
}

void
TiogaBlock::adjust_cell_resolutions()
{
  // For every face on the sideset, grab the connected element and set its
  // cell resolution to a large value. Also for each node of that element, set
  // the nodal resolution to a large value.

  constexpr double large_volume = std::numeric_limits<double>::max();
  stk::mesh::Selector mesh_selector = get_node_selector(ovsetParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);
  auto* nodeVol =
    meta_.get_field<double>(stk::topology::NODE_RANK, "tioga_nodal_volume");

  auto& eidmap = bdata_.eid_map_.h_view;
  auto& cellres = bdata_.cell_res_.h_view;

  for (auto b : mbkts) {
    for (const auto node : *b) {
      const auto* elems = bulk_.begin_elements(node);
      const auto num_elems = bulk_.num_elements(node);

      for (unsigned ie = 0; ie < num_elems; ++ie) {
        const auto elem = elems[ie];
        const int eidx = eidmap(elem.local_offset()) - 1;
        cellres[eidx] = large_volume;

        const auto* elem_nodes = bulk_.begin_nodes(elem);
        const auto num_enodes = bulk_.num_nodes(elem);
        for (unsigned ii = 0; ii < num_enodes; ++ii) {
          double* vol = stk::mesh::field_data(*nodeVol, elem_nodes[ii]);
          vol[0] = large_volume;
        }
      }
    }
  }

  bdata_.cell_res_.sync_device();
}

void
TiogaBlock::adjust_node_resolutions()
{
  stk::mesh::Selector mesh_selector = get_node_selector(blkParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);
  sierra::nalu::ScalarFieldType* nodeVol =
    meta_.get_field<double>(stk::topology::NODE_RANK, "tioga_nodal_volume");

  auto& eidmap = bdata_.eid_map_.h_view;
  auto& noderes = bdata_.node_res_.h_view;

  for (auto b : mbkts) {
    const double* nVol = stk::mesh::field_data(*nodeVol, *b);
    for (size_t in = 0; in < b->size(); ++in) {
      const auto node = (*b)[in];
      const int nidx = eidmap(node.local_offset()) - 1;
      noderes[nidx] = nVol[in];
    }
  }

  bdata_.node_res_.sync_device();
}

void
TiogaBlock::get_donor_info(TIOGA::tioga& tg, stk::mesh::EntityProcVec& egvec)
{
  // Do nothing if this mesh block isn't present in this MPI Rank
  if (num_nodes_ < 1)
    return;

  int dcount, fcount;

  // Call TIOGA API to determine donor info array sizes
  tg.getDonorCount(meshtag_, &dcount, &fcount);

  // Receptor info: rProcID, rNodeID, blkID, nFractions
  std::vector<int> receptorInfo(dcount * 4);
  // Node index information (the last entry is the donor element ID)
  std::vector<int> inode(fcount);
  // fractions (ignored for now). This is useful if we want TIOGA to handle
  // field interpolations. In Nalu, we will use STK + master_element calls to
  // perform this without TIOGA's help.
  std::vector<double> frac(fcount);

  // Populate the donor information arrays through TIOGA API call
  tg.getDonorInfo(
    meshtag_, receptorInfo.data(), inode.data(), frac.data(), &dcount);

  // With getDonorInfo TIOGA returns information about the donor elements (in
  // the current MPI rank) that are providing information to receptor nodes
  // belonging to another mesh. The integer array returned from this method
  // contains 4 entries per {receptor node, donor element} pair.
  //
  //   - The MPI rank of the receptor node
  //   - The receptor node id (local index into tioga array)
  //   - The mesh tag for the receptor node ID
  //   - The topology.num_nodes() + 1 of the donor element
  //
  // For ghosting elements, we only need the first and the last entry to be
  // processed within this method.
  int myRank = bulk_.parallel_rank();
  int idx = 0;
  for (int i = 0; i < (4 * dcount); i += 4) {
    int procid = receptorInfo[i];
    int nweights = receptorInfo[i + 3];     // Offset to get the donor element
    int elemid_tmp = inode[idx + nweights]; // Local index for lookup
    auto elemID = bdata_.cell_gid_.h_view[elemid_tmp]; // Global ID of element

    // Move the offset index for next call
    idx += nweights + 1;

    // No ghosting necessary if sharing the same rank
    if (procid == myRank)
      continue;

    // Add this donor element to the elementsToGhost vector
    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, elemID);
    stk::mesh::EntityProc elem_proc(elem, procid);
    egvec.push_back(elem_proc);
  }
}

inline void
TiogaBlock::names_to_parts(
  const std::vector<std::string>& pnames, stk::mesh::PartVector& parts)
{
  parts.resize(pnames.size());
  for (size_t i = 0; i < pnames.size(); i++) {
    stk::mesh::Part* p = meta_.get_part(pnames[i]);
    if (nullptr == p) {
      throw std::runtime_error(
        "TiogaBlock: cannot find part named: " + pnames[i]);
    } else {
      parts[i] = p;
    }
  }
}

stk::mesh::Selector
TiogaBlock::get_node_selector(stk::mesh::PartVector& parts)
{
  return stk::mesh::selectUnion(parts) &
         (meta_.locally_owned_part() | meta_.globally_shared_part());
}

stk::mesh::Selector
TiogaBlock::get_elem_selector(stk::mesh::PartVector& parts)
{
  return stk::mesh::selectUnion(parts) & meta_.locally_owned_part();
}

void
TiogaBlock::process_nodes()
{
  stk::mesh::Selector mesh_selector = get_node_selector(blkParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);
  sierra::nalu::VectorFieldType* coords =
    meta_.get_field<double>(stk::topology::NODE_RANK, coords_name_);
  sierra::nalu::ScalarFieldType* nodeVol =
    meta_.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");

  int ncount = 0;
  for (auto b : mbkts)
    ncount += b->size();

  if (is_init_ || ncount != num_nodes_) {
    num_nodes_ = ncount;

    bdata_.xyz_.init("xyz", ndim_ * num_nodes_);
    bdata_.iblank_.init("iblank_node", num_nodes_);
    bdata_.node_res_.init("node_res", num_nodes_);
    bdata_.eid_map_.init(
      "stk_to_tioga_id", bulk_.get_size_of_entity_index_space());
    bdata_.node_gid_.init("node_gid", num_nodes_);
  }

  const double fac = tiogaOpts_.node_res_mult();
  auto& ngp_xyz = bdata_.xyz_.h_view;
  auto& nidmap = bdata_.eid_map_.h_view;
  auto& nodegid = bdata_.node_gid_.h_view;
  auto& node_res = bdata_.node_res_.h_view;
  int ip = 0; // Index into the xyz_ array
  for (auto b : mbkts) {
    for (size_t in = 0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];
      stk::mesh::EntityId nid = bulk_.identifier(node);

      double* pt = stk::mesh::field_data(*coords, node);
      double* nVol = stk::mesh::field_data(*nodeVol, node);
      for (int i = 0; i < ndim_; i++) {
        ngp_xyz(ip * ndim_ + i) = pt[i];
      }

      node_res(ip) = *nVol * fac;
      nidmap(node.local_offset()) = ip + 1; // TIOGA uses 1-based indexing
      nodegid(ip) = nid;
      ip++;
    }
  }

  bdata_.xyz_.sync_device();
  bdata_.node_res_.sync_device();
  bdata_.eid_map_.sync_device();
  bdata_.node_gid_.sync_device();
  Kokkos::deep_copy(bdata_.iblank_.h_view, 1);
  Kokkos::deep_copy(bdata_.iblank_.d_view, 1);
}

void
TiogaBlock::process_wallbc()
{
  stk::mesh::Selector mesh_selector = get_node_selector(wallParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);

  int ncount = 0;
  for (auto b : mbkts)
    ncount += b->size();

  if (is_init_ || (ncount != num_wallbc_)) {
    num_wallbc_ = ncount;
    bdata_.wallIDs_.init("wall_ids", num_wallbc_);
  }

  auto& wallids = bdata_.wallIDs_.h_view;
  auto& nidmap = bdata_.eid_map_.h_view;
  int ip = 0; // Index into the wallIDs array
  for (auto b : mbkts) {
    for (size_t in = 0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];
      wallids(ip++) = nidmap(node.local_offset());
    }
  }
  bdata_.wallIDs_.sync_device();
}

void
TiogaBlock::process_ovsetbc()
{
  stk::mesh::Selector mesh_selector = get_node_selector(ovsetParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);

  int ncount = 0;
  for (auto b : mbkts)
    ncount += b->size();

  if (is_init_ || (ncount != num_ovsetbc_)) {
    num_ovsetbc_ = ncount;
    bdata_.ovsetIDs_.init("overset_ids", num_ovsetbc_);
  }

  auto& ovsetids = bdata_.ovsetIDs_.h_view;
  auto& nidmap = bdata_.eid_map_.h_view;
  int ip = 0; // Index into ovsetIDs array
  for (auto b : mbkts) {
    for (size_t in = 0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];
      ovsetids(ip++) = nidmap(node.local_offset());
    }
  }
  bdata_.ovsetIDs_.sync_device();
}

void
TiogaBlock::process_elements()
{
  stk::mesh::Selector mesh_selector = get_elem_selector(blkParts_);
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::ELEM_RANK, mesh_selector);

  // 1. Determine the number of topologies present in this mesh block. For
  // each topology determine the number of elements associated with it (across
  // all buckets). We will use this for resizing arrays later on.
  for (auto b : mbkts) {
    size_t num_elems = b->size();
    // npe = Nodes Per Elem
    int npe = b->topology().num_nodes();
    auto topo = conn_map_.find(npe);
    if (topo != conn_map_.end()) {
      conn_map_[npe] += num_elems;
    } else {
      conn_map_[npe] = num_elems;
    }
  }

  // 2. Resize arrays used to pass data to TIOGA grid registration interface
  auto ntypes = conn_map_.size();
  bdata_.num_verts_.init("num_verts_per_etype", ntypes);
  bdata_.num_cells_.init("num_cells_per_etype", ntypes);

  if (tioga_conn_)
    delete[] tioga_conn_;
  tioga_conn_ = new int*[ntypes];

  std::map<int, int> conn_ids;        // Topo -> array index lookup table
  std::map<int, size_t> conn_offsets; // Topo -> array offset lookup table
  std::map<int, size_t> elem_offsets;

  // 3. Populate TIOGA data structures
  int idx = 0;
  size_t eoffset = 0;
  size_t tot_elems = 0;
  for (auto kv : conn_map_) {
    tot_elems += kv.second;
    {
      bdata_.num_verts_.h_view(idx) = kv.first;
      bdata_.num_cells_.h_view(idx) = kv.second;
      bdata_.connect_[idx].init(
        "cell_" + std::to_string(kv.first), kv.first * kv.second);
    }
    conn_ids[kv.first] = idx;
    conn_offsets[kv.first] = 0;
    elem_offsets[kv.first] = eoffset;
    idx++;
    eoffset += kv.second;
  }

  bdata_.iblank_cell_.init("iblank_cell", tot_elems);
  bdata_.cell_res_.init("cell_res", tot_elems);
  bdata_.cell_gid_.init("cell_gid", tot_elems);

  // 4. Create connectivity map based on local node index (xyz_)
  auto& eidmap = bdata_.eid_map_.h_view;
  auto& cellgid = bdata_.cell_gid_.h_view;
  for (auto b : mbkts) {
    const int npe = b->num_nodes(0);
    const int idx = conn_ids[npe];
    int offset = conn_offsets[npe];
    int ep = elem_offsets[npe];
    for (size_t in = 0; in < b->size(); in++) {
      const stk::mesh::Entity elem = (*b)[in];
      const stk::mesh::EntityId eid = bulk_.identifier(elem);
      eidmap(elem.local_offset()) = ep + 1;
      cellgid(ep++) = eid;
      const stk::mesh::Entity* enodes = b->begin_nodes(in);
      for (int i = 0; i < npe; i++) {
        bdata_.connect_[idx].h_view(offset++) =
          eidmap(enodes[i].local_offset());
      }
    }
    conn_offsets[npe] = offset;
    elem_offsets[npe] = ep;
  }

  // TIOGA expects a ptr-to-ptr data structure for connectivity
  for (size_t i = 0; i < ntypes; i++) {
    tioga_conn_[i] = bdata_.connect_[i].h_view.data();
  }

  bdata_.eid_map_.sync_device();
  bdata_.cell_gid_.sync_device();
  bdata_.num_verts_.sync_device();
  bdata_.num_cells_.sync_device();

  Kokkos::deep_copy(bdata_.iblank_cell_.h_view, 1);
  Kokkos::deep_copy(bdata_.iblank_cell_.d_view, 1);
}

void
TiogaBlock::reset_iblank_data()
{
  Kokkos::deep_copy(bdata_.iblank_.h_view, 1);
  Kokkos::deep_copy(bdata_.iblank_.d_view, 1);
  Kokkos::deep_copy(bdata_.iblank_cell_.h_view, 1);
  Kokkos::deep_copy(bdata_.iblank_cell_.d_view, 1);
}

void
TiogaBlock::register_block(TIOGA::tioga& tg)
{
  // Do nothing if this mesh block isn't present in this MPI Rank
  if (num_nodes_ < 1)
    return;

  // Reset iblanks before tioga connectivity
  reset_iblank_data();

  // Register the mesh block information to TIOGA
  tg.registerGridData(
    meshtag_,                        // Unique body tag
    num_nodes_,                      // Number of nodes in this mesh block
    bdata_.xyz_.h_view.data(),       // Nodal coordinates
    bdata_.iblank_.h_view.data(),    // iblank array corresponding to nodes
    num_wallbc_,                     // Number of Wall BC nodes
    num_ovsetbc_,                    // Number of overset BC nodes
    bdata_.wallIDs_.h_view.data(),   // Node IDs of wall BC nodes
    bdata_.ovsetIDs_.h_view.data(),  // Node IDs of overset BC nodes
    bdata_.num_verts_.h_view.size(), // Number of topologies in this mesh block
    bdata_.num_verts_.h_view.data(), // Number of vertices per topology
    bdata_.num_cells_.h_view.data(), // Number of cells for each topology
    tioga_conn_,                     // Element node connectivity information
    bdata_.cell_gid_.h_view.data()   // Global ID for the element array
#ifdef TIOGA_HAS_NODEGID
    ,
    bdata_.node_gid_.h_view.data() // Global ID for the node array
#endif
  );
  // Indicate that we want element IBLANK information returned
  tg.set_cell_iblank(meshtag_, bdata_.iblank_cell_.h_view.data());

  // Register cell/node resolutions for TIOGA
  if (tiogaOpts_.set_resolutions())
    tg.setResolutions(
      meshtag_, bdata_.node_res_.h_view.data(), bdata_.cell_res_.h_view.data());
}

void
TiogaBlock::print_summary()
{
  std::vector<double> bboxMin(ndim_, std::numeric_limits<double>::max());
  std::vector<double> bboxMax(ndim_, -std::numeric_limits<double>::max());
  stk::mesh::EntityId nidMin = std::numeric_limits<unsigned>::max();
  stk::mesh::EntityId nidMax = 0;

  auto& xyz = bdata_.xyz_.h_view;
  auto& nodegid = bdata_.node_gid_.h_view;
  for (int i = 0; i < num_nodes_; i++) {
    nidMin = std::min(nodegid[i], nidMin);
    nidMax = std::max(nodegid[i], nidMax);

    for (int j = 0; j < ndim_; j++) {
      int k = i * 3 + j;
      bboxMax[j] = std::max(xyz[k], bboxMax[j]);
      bboxMin[j] = std::min(xyz[k], bboxMin[j]);
    }
  }

  std::vector<double> gboxMin(ndim_);
  std::vector<double> gboxMax(ndim_);
  stk::mesh::EntityId gnidMin, gnidMax;
  stk::all_reduce_min(bulk_.parallel(), bboxMin.data(), gboxMin.data(), ndim_);
  stk::all_reduce_max(bulk_.parallel(), bboxMax.data(), gboxMax.data(), ndim_);
  stk::all_reduce_min(bulk_.parallel(), &nidMin, &gnidMin, 1);
  stk::all_reduce_max(bulk_.parallel(), &nidMax, &gnidMax, 1);

  sierra::nalu::NaluEnv::self().naluOutputP0()
    << "TIOGA: mesh block = " << block_name_ << "; ID min = " << gnidMin
    << "; ID max = " << gnidMax << "\n"
    << "\tBounding box: \n\t\t[" << gboxMin[0] << ", " << gboxMin[1] << ", "
    << gboxMin[2] << "]\n\t\t[" << gboxMax[0] << ", " << gboxMax[1] << ", "
    << gboxMax[2] << "]\n"
    << std::endl;
}

void
TiogaBlock::register_solution(
  TIOGA::tioga& tg,
  const std::vector<sierra::nalu::OversetFieldData>& fields,
  const int ncomp)
{
  if (num_nodes_ < 1)
    return;

  size_t num_field_entries = num_nodes_ * ncomp;
  auto& qsol = bdata_.qsol_;
  if (qsol.size() < num_field_entries)
    qsol.init("stk_soln_array", num_field_entries);

  const stk::mesh::Selector sel = get_node_selector(blkParts_);
  const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

  auto& qsolarr = qsol.h_view;
  size_t idx = 0;
  for (auto* b : bkts) {
    for (size_t in = 0; in < b->size(); ++in) {
      auto node = (*b)[in];

      for (auto& finfo : fields) {
        auto* fld = finfo.field_;
        const size_t fsize = finfo.sizeRow_ * finfo.sizeCol_;

        double* fdata = static_cast<double*>(stk::mesh::field_data(*fld, node));
        for (size_t ic = 0; ic < fsize; ++ic)
          qsolarr(idx++) = fdata[ic];
      }
    }
  }

#if TIOGA_HAS_NGP_IFACE
  constexpr int row_major = 0;
  tg.register_unstructured_solution(
    meshtag_, qsol.h_view.data(), ncomp, row_major);
#else
  tg.registerSolution(meshtag_, qsol.h_view.data());
#endif
}

void
TiogaBlock::register_solution(
  TIOGA::tioga& tg, const sierra::nalu::OversetFieldData& field)
{
  if (num_nodes_ < 1)
    return;

  const size_t fsize = field.sizeRow_ * field.sizeCol_;
  const size_t num_field_entries = num_nodes_ * fsize;
  auto& qsol = bdata_.qsol_;
  if (qsol.size() < num_field_entries)
    qsol.init("stk_soln_array", num_field_entries);

  const stk::mesh::Selector sel = get_node_selector(blkParts_);
  const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);
  const auto* fld = field.field_;

  auto& qsolarr = qsol.h_view;
  size_t idx = 0;
  for (auto* b : bkts) {
    double* fdata = static_cast<double*>(stk::mesh::field_data(*fld, *b));

    for (size_t in = 0; in < b->size(); ++in) {
      for (size_t ic = 0; ic < fsize; ++ic) {
        qsolarr(idx++) = fdata[in * fsize + ic];
      }
    }
  }

#if TIOGA_HAS_NGP_IFACE
  constexpr int row_major = 0;
  tg.register_unstructured_solution(
    meshtag_, qsol.h_view.data(), fsize, row_major);
#else
  tg.registerSolution(meshtag_, qsol.h_view.data());
#endif
}

void
TiogaBlock::update_solution(
  const std::vector<sierra::nalu::OversetFieldData>& fields)
{
  if (num_nodes_ < 1)
    return;

  const stk::mesh::Selector sel = get_node_selector(blkParts_);
  const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

  const auto& qsolarr = bdata_.qsol_.h_view;
  size_t idx = 0;
  for (auto* b : bkts) {
    for (size_t in = 0; in < b->size(); ++in) {
      auto node = (*b)[in];

      for (auto& finfo : fields) {
        auto* fld = finfo.field_;
        const size_t fsize = finfo.sizeRow_ * finfo.sizeCol_;

        double* fdata = static_cast<double*>(stk::mesh::field_data(*fld, node));
        for (size_t ic = 0; ic < fsize; ++ic)
          fdata[ic] = qsolarr(idx++);
      }
    }
  }
}

void
TiogaBlock::update_solution(const sierra::nalu::OversetFieldData& field)
{
  if (num_nodes_ < 1)
    return;

  const stk::mesh::Selector sel = get_node_selector(blkParts_);
  const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);
  const auto* fld = field.field_;
  const size_t fsize = field.sizeRow_ * field.sizeCol_;

  const auto& qsolarr = bdata_.qsol_.h_view;
  size_t idx = 0;
  for (auto* b : bkts) {
    double* fdata = static_cast<double*>(stk::mesh::field_data(*fld, *b));

    for (size_t in = 0; in < b->size(); ++in) {
      for (size_t ic = 0; ic < fsize; ++ic) {
        fdata[in * fsize + ic] = qsolarr(idx++);
      }
    }
  }
}

} // namespace tioga_nalu

#endif // NALU_USES_TIOGA
