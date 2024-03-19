// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FromMesh_h
#define FromMesh_h

#include <string>
#include <vector>
#include <utility>

#include <Realm.h>
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <stk_search/IdentProc.hpp>
#include <stk_search/BoundingBox.hpp>

#include <FieldTypeDef.h>

// stk
namespace stk {
namespace mesh {
class Part;
typedef std::vector<Part*> PartVector;
class MetaData;
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class FromMesh
{
public:
  typedef stk::mesh::Entity Entity;
  typedef std::vector<Entity> EntityVec;
  typedef stk::mesh::EntityKey EntityKey;
  typedef std::set<EntityKey> EntityKeySet;
  typedef stk::search::IdentProc<EntityKey, unsigned> EntityProc;
  typedef std::vector<EntityProc> EntityProcVec;

  typedef stk::search::Point<double> Point;
  typedef stk::search::Box<double> Box;
  typedef std::pair<Box, EntityProc> BoundingBox;

  enum { Dimension = 3 };

  typedef std::vector<std::pair<std::string, std::string>> PairNames;

  std::vector<const stk::mesh::FieldBase*> get_fields(
    const stk::mesh::MetaData& fromMetaData, const PairNames& VarPairName)
  {
    // will want to check that all is well with field registration
    bool allFieldsAreFine = true;
    std::vector<const stk::mesh::FieldBase*> fromFieldVec;
    // provide field names
    for (PairNames::const_iterator i = VarPairName.begin();
         i != VarPairName.end(); ++i) {
      const std::string& fieldName = i->first;
      const stk::mesh::FieldBase* fromfield =
        stk::mesh::get_field_by_name(fieldName, fromMetaData);
      if (NULL == fromfield) {
        allFieldsAreFine = false;
        NaluEnv::self().naluOutputP0()
          << "Xfer::FromMesh:Error field: " << fieldName
          << " has not been registered anywhere within the FromRealm: "
          << fromRealm_.name() << std::endl;
      } else {
        // always push back; check for errors below
        fromFieldVec.push_back(fromfield);

        // check that the field is defined on **all** parts
        stk::mesh::Selector fieldSelector = stk::mesh::selectField(*fromfield);
        for (size_t k = 0; k < fromPartVec_.size(); ++k) {

          stk::mesh::BucketVector const& partBuckets =
            fromBulkData_.get_buckets(
              stk::topology::NODE_RANK, stk::mesh::Selector(*fromPartVec_[k]));

          bool fieldIsFine = true;
          for (stk::mesh::BucketVector::const_iterator ib = partBuckets.begin();
               ib != partBuckets.end(); ++ib) {
            stk::mesh::Bucket& b = **ib;
            fieldIsFine &= fieldSelector(b);
          }

          // local check to make sure that the field is somewhere (delay the
          // throw)
          if (!fieldIsFine) {
            NaluEnv::self().naluOutputP0()
              << "Xfer::FromMesh:Error field: " << fromfield->name()
              << " is not registered on part: " << fromPartVec_[k]->name()
              << std::endl;
            allFieldsAreFine = false;
          }
        }
      }
    }

    // final error check; only return when all is well
    if (allFieldsAreFine) {
      return fromFieldVec;
    } else {
      throw std::runtime_error("Xfer::FromMesh:Error field registration on "
                               "desired parts of the mesh is not complete");
    }
  }

  FromMesh(
    const stk::mesh::MetaData& fromMetaData,
    stk::mesh::BulkData& fromBulkData,
    Realm& fromRealm,
    const std::string& coordinates_name,
    const PairNames& VarPairName,
    const stk::mesh::PartVector& fromPartVec,
    const stk::ParallelMachine comm)
    : fromMetaData_(fromMetaData),
      fromBulkData_(fromBulkData),
      fromRealm_(fromRealm),
      fromcoordinates_(fromMetaData.get_field<double>(
        stk::topology::NODE_RANK, coordinates_name)),
      fromPartVec_(fromPartVec),
      fromFieldVec_(get_fields(fromMetaData, VarPairName)),
      comm_(comm),
      mesh_modified_(false),
      ghosting_(0),
      ghosting_map_()
  {
    // nothing to do
  }

  ~FromMesh(){};

  struct BoundingBoxCompare
  {
    bool operator()(const BoundingBox& a, const BoundingBox& b) const
    {
      return a.second.id() < b.second.id();
    }
  };

  // Needed for STK Transfer
  stk::ParallelMachine comm() const { return comm_; }

  void bounding_boxes(std::vector<BoundingBox>& v) const
  {
    const unsigned nDim = fromMetaData_.spatial_dimension();

    Point min_corner, max_corner;

    stk::mesh::Selector s_locally_owned_union =
      fromMetaData_.locally_owned_part() & stk::mesh::selectUnion(fromPartVec_);

    // determine entity rank for the part served up; should be homogeneous
    stk::mesh::EntityRank partEntityRank =
      fromPartVec_[0]->primary_entity_rank();

    stk::mesh::BucketVector const& entity_buckets =
      fromBulkData_.get_buckets(partEntityRank, s_locally_owned_union);
    for (stk::mesh::BucketVector::const_iterator ib = entity_buckets.begin();
         ib != entity_buckets.end(); ++ib) {
      stk::mesh::Bucket& b = **ib;

      const stk::mesh::Bucket::size_type length = b.size();

      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

        // get entity
        stk::mesh::Entity theEntity = b[k];

        // initialize max and min
        for (unsigned j = 0; j < nDim; ++j) {
          min_corner[j] = +1.0e16;
          max_corner[j] = -1.0e16;
        }

        stk::mesh::Entity const* entity_node_rels =
          fromBulkData_.begin_nodes(theEntity);
        int num_entity_nodes = fromBulkData_.num_nodes(theEntity);
        for (int ni = 0; ni < num_entity_nodes; ++ni) {
          stk::mesh::Entity node = entity_node_rels[ni];

          double* coords = stk::mesh::field_data(*fromcoordinates_, node);
          for (unsigned j = 0; j < nDim; ++j) {
            min_corner[j] = std::min(min_corner[j], coords[j]);
            max_corner[j] = std::max(max_corner[j], coords[j]);
          }
        }

        // setup ident
        FromMesh::EntityProc theIdent(
          fromBulkData_.entity_key(theEntity), fromBulkData_.parallel_rank());

        v.push_back(BoundingBox(Box(min_corner, max_corner), theIdent));
      }
    }
    std::sort(v.begin(), v.end(), BoundingBoxCompare());
  }

  void update_ghosting(const EntityProcVec& entity_keys)
  {
    ghosting_map_.resize(entity_keys.size());
    for (size_t i = 0; i < entity_keys.size(); ++i) {
      // convert from EntityProc based on EntityKey to EntityProc based on raw
      // Entity.
      const EntityProc& key_proc = entity_keys[i];
      const stk::mesh::EntityKey key = key_proc.id();
      const unsigned proc = key_proc.proc();
      const stk::mesh::Entity e = entity(key);
      const stk::mesh::EntityProc ep(e, proc);
      ghosting_map_[i] = ep;
    }

    unsigned s = !ghosting_map_.empty();
    stk::all_reduce(comm_, stk::ReduceSum<1>(&s));

    if (s) {
      std::sort(ghosting_map_.begin(), ghosting_map_.end());
      stk::mesh::EntityProcVec::iterator del =
        std::unique(ghosting_map_.begin(), ghosting_map_.end());
      ghosting_map_.resize(std::distance(ghosting_map_.begin(), del));

      std::string theGhostName = "nalu_transfer_ghosting";
      for (unsigned i = 0; i != fromFieldVec_.size(); ++i)
        theGhostName += "_" + fromFieldVec_[i]->name();
      ghosting_ = &fromBulkData_.create_ghosting(theGhostName);
      fromBulkData_.change_ghosting(*ghosting_, ghosting_map_);
      mesh_modified_ = true;
    }
  }

  void update_values()
  {
    if (ghosting_) {
      std::vector<const stk::mesh::FieldBase*> fields(
        fromFieldVec_.begin(), fromFieldVec_.end());
      if (mesh_modified_) {
        // Copy coordinates to the newly ghosted nodes
        mesh_modified_ = false;
        fields.push_back(fromcoordinates_);
      }
      stk::mesh::communicate_field_data(*ghosting_, fields);
      stk::mesh::copy_owned_to_shared(fromBulkData_, fields);
    }
  }

  Entity entity(const EntityKey k) const { return fromBulkData_.get_entity(k); }

  const stk::mesh::MetaData& fromMetaData_;
  stk::mesh::BulkData& fromBulkData_;
  Realm& fromRealm_;
  const VectorFieldType* fromcoordinates_;
  const stk::mesh::PartVector fromPartVec_;
  const std::vector<const stk::mesh::FieldBase*> fromFieldVec_;
  const stk::ParallelMachine comm_;

  bool mesh_modified_;
  stk::mesh::Ghosting* ghosting_;
  stk::mesh::EntityProcVec ghosting_map_;
};

} // namespace nalu
} // namespace sierra

#endif
