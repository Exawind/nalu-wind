// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef OVERSETMANAGER_H
#define OVERSETMANAGER_H

#include "overset/OversetFieldData.h"

#include <stk_mesh/base/Selector.hpp>

#include <vector>

namespace stk {
namespace io {
class StkMeshIoBroker;
}

namespace mesh {
class Part;
class MetaData;
class BulkData;
class Ghosting;
class FieldBase;
typedef std::vector<Part*> PartVector;
struct Entity;
}
}

namespace sierra {
namespace nalu {

class Realm;
class OversetInfo;

/** Base class for Overset connectivity manager
 *
 */
class OversetManager
{
public:
  OversetManager(Realm& realm);

  virtual ~OversetManager();

  /** Deallocate OversetInfo memory allocated via new upoin reinitialization */
  void delete_info_vec();

  /** Perform any necessary setup actions for overset algorithms
   *
   *  Part and field registration to STK
   */
  virtual void setup();

  /** Setup all data structures and perform connectivity
   *
   * This method must be implemented by concrete OGA implementations.
   */
  virtual void initialize(const bool isDecoupled = false) = 0;

  virtual void overset_orphan_node_field_update(
    stk::mesh::FieldBase*,
    const int,
    const int);

  /** Return an inactive selector that contains the hole elements
   */
  virtual stk::mesh::Selector get_inactive_selector();

  virtual void overset_update_fields(const std::vector<OversetFieldData>&) = 0;

  virtual void overset_update_field(
    stk::mesh::FieldBase* field, int nrows = 1, int ncols = 1) = 0;

  Realm& realm_;

  stk::mesh::MetaData* metaData_{nullptr};

  stk::mesh::BulkData* bulkData_{nullptr};

  stk::mesh::Ghosting* oversetGhosting_{nullptr};

  std::vector<OversetInfo*> oversetInfoVec_;

  std::vector<stk::mesh::Entity> holeNodes_;
  std::vector<stk::mesh::Entity> fringeNodes_;

  std::vector<int> ghostCommProcs_;

private:
  OversetManager() = delete;
  OversetManager(const OversetManager&) = delete;

};

} // nalu
} // sierra

#endif /* OVERSETMANAGER_H */
