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

#include "KokkosInterface.h"
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
  using EntityList = Kokkos::View<stk::mesh::Entity*, Kokkos::LayoutRight, MemSpace>;

  OversetManager(Realm& realm);

  virtual ~OversetManager();

  /** Deallocate OversetInfo memory allocated via new upoin reinitialization */
  void delete_info_vec();

  /** Perform any necessary setup actions for overset algorithms
   *
   *  Part and field registration to STK
   */
  virtual void setup();

  /** Setup all the initial data structures (one time setup)
   */
  virtual void initialize() = 0;

  /** Perform overset connectivity
   */
  virtual void execute(const bool isDecoupled) = 0;

  virtual void overset_orphan_node_field_update(
    stk::mesh::FieldBase*,
    const int,
    const int);

  /** Return an inactive selector that contains the hole elements
   */
  virtual stk::mesh::Selector get_inactive_selector();

  virtual void overset_update_fields(const std::vector<OversetFieldData>&) = 0;

  virtual void overset_update_field(
    stk::mesh::FieldBase* field, const int nrows = 1, const int ncols = 1,
    const bool doFinalSyncToDevice = true) = 0;

  virtual void reset_data_structures();

  Realm& realm_;

  stk::mesh::MetaData* metaData_{nullptr};

  stk::mesh::BulkData* bulkData_{nullptr};

  stk::mesh::Ghosting* oversetGhosting_{nullptr};

  std::vector<OversetInfo*> oversetInfoVec_;

  std::vector<stk::mesh::Entity> holeNodes_;
  std::vector<stk::mesh::Entity> fringeNodes_;

  EntityList ngpHoleNodes_;
  EntityList ngpFringeNodes_;

  std::vector<int> ghostCommProcs_;

  //! Timer for overset connectivity
  double timerConnectivity_{0.0};

  //! Timer for overset field interpolations
  double timerFieldUpdate_{0.0};

private:
  OversetManager() = delete;
  OversetManager(const OversetManager&) = delete;

};

} // nalu
} // sierra

#endif /* OVERSETMANAGER_H */
