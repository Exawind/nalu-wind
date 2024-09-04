// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ToMesh_h
#define ToMesh_h

#include <string>
#include <vector>
#include <utility>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>

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

class ToMesh
{
public:
  typedef stk::mesh::Entity Entity;
  typedef std::vector<Entity> EntityVec;
  typedef stk::mesh::EntityKey EntityKey;
  typedef std::set<EntityKey> EntityKeySet;
  typedef stk::search::IdentProc<EntityKey, unsigned> EntityProc;
  typedef std::vector<EntityProc> EntityProcVec;

  typedef stk::search::Point<double> Point;
  typedef stk::search::Sphere<double> Sphere;
  typedef std::pair<Sphere, EntityProc> BoundingBox;

  enum { Dimension = 3 };

  typedef std::vector<std::pair<std::string, std::string>> PairNames;
  std::vector<const stk::mesh::FieldBase*> get_fields(
    const stk::mesh::MetaData& toMetaData, const PairNames& VarPairName)
  {
    // will want to check that all is well with field registration
    bool allFieldsAreFine = true;
    std::vector<const stk::mesh::FieldBase*> toFieldVec;
    // provide field names
    for (PairNames::const_iterator i = VarPairName.begin();
         i != VarPairName.end(); ++i) {
      const std::string& fieldName = i->second;
      const stk::mesh::FieldBase* tofield =
        stk::mesh::get_field_by_name(fieldName, toMetaData);
      if (NULL == tofield) {
        allFieldsAreFine = false;
        NaluEnv::self().naluOutputP0()
          << "Xfer::ToMesh:Error field: " << fieldName
          << " has not been registered anywhere within the ToRealm: "
          << toRealm_.name() << std::endl;
      } else {
        // always push back; check for errors below
        toFieldVec.push_back(tofield);

        // check that the field is defined on **all** parts
        stk::mesh::Selector fieldSelector = stk::mesh::selectField(*tofield);
        for (size_t k = 0; k < toPartVec_.size(); ++k) {

          const stk::mesh::BucketVector& partBuckets = toBulkData_.get_buckets(
            stk::topology::NODE_RANK, stk::mesh::Selector(*toPartVec_[k]));

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
              << "Xfer::ToMesh:Error field: " << tofield->name()
              << " is not registered on part: " << toPartVec_[k]->name()
              << std::endl;
            allFieldsAreFine = false;
          }
        }
      }
    }

    // final error check; only return when all is well
    if (allFieldsAreFine) {
      return toFieldVec;
    } else {
      throw std::runtime_error("Xfer::ToMesh:Error field registration on "
                               "desired parts of the mesh is not complete");
    }
  }

  ToMesh(
    stk::mesh::MetaData& toMetaData,
    stk::mesh::BulkData& toBulkData,
    Realm& toRealm,
    const std::string& coordinates_name,
    const PairNames& VarPairName,
    const stk::mesh::PartVector& toPartVec,
    const stk::ParallelMachine comm,
    const double radius)
    : toMetaData_(toMetaData),
      toBulkData_(toBulkData),
      toRealm_(toRealm),
      tocoordinates_(toMetaData.get_field<double>(
        stk::topology::NODE_RANK, coordinates_name)),
      toPartVec_(toPartVec),
      toFieldVec_(get_fields(toMetaData, VarPairName)),
      comm_(comm),
      radius_(radius)
  {
    // nothing to do
  }

  ~ToMesh() {};

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

    const unsigned spatial_dimension = toMetaData_.spatial_dimension();

    Point center;

    stk::mesh::Selector s_locally_owned_union =
      toMetaData_.locally_owned_part() & stk::mesh::selectUnion(toPartVec_);

    stk::mesh::BucketVector const& node_buckets =
      toBulkData_.get_buckets(stk::topology::NODE_RANK, s_locally_owned_union);
    for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
         ib != node_buckets.end(); ++ib) {
      stk::mesh::Bucket& b = **ib;

      const stk::mesh::Bucket::size_type length = b.size();

      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

        // get node
        stk::mesh::Entity node = b[k];

        double* coord = stk::mesh::field_data(*tocoordinates_, node);
        for (unsigned i = 0; i < spatial_dimension; ++i) {
          center[i] = coord[i];
        }

        // setup ident
        ToMesh::EntityProc theIdent(
          toBulkData_.entity_key(node), toBulkData_.parallel_rank());

        ToMesh::BoundingBox theBox(Sphere(center, radius_), theIdent);
        v.push_back(theBox);
      }
    }
    std::sort(v.begin(), v.end(), BoundingBoxCompare());
  }

  void update_values()
  {
    std::vector<const stk::mesh::FieldBase*> fields(
      toFieldVec_.begin(), toFieldVec_.end());
    stk::mesh::copy_owned_to_shared(toBulkData_, fields);
  }

  stk::mesh::MetaData& toMetaData_;
  stk::mesh::BulkData& toBulkData_;
  Realm& toRealm_;
  const VectorFieldType* tocoordinates_;
  const stk::mesh::PartVector toPartVec_;
  const std::vector<const stk::mesh::FieldBase*> toFieldVec_;
  const stk::ParallelMachine comm_;
  const double radius_;

  typedef std::map<stk::mesh::EntityKey, std::vector<double>> TransferInfo;
  TransferInfo TransferInfo_;
};

} // namespace nalu
} // namespace sierra

#endif
