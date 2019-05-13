/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <master_element/MasterElement.h>
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

#include <stk_util/parallel/ParallelReduce.hpp>

// vector and pair
#include <vector>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// Acon_NonConformalManager - manages nonConformal info
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
NonConformalManager::NonConformalManager(
  Realm &realm,
  const bool ncAlgDetailedOutput,
  const bool ncAlgCoincidentNodesErrorCheck)
  : realm_(realm ),
    ncAlgDetailedOutput_(ncAlgDetailedOutput),
    ncAlgCoincidentNodesErrorCheck_(ncAlgCoincidentNodesErrorCheck),
    nonConformalGhosting_(NULL)
{
  // do nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
NonConformalManager::~NonConformalManager()
{
  // delete nonConformal info objects:
  std::vector<NonConformalInfo *>::iterator ic;
  for( ic=nonConformalInfoVec_.begin();
       ic!=nonConformalInfoVec_.end(); ++ic )
    delete (*ic);
}


//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalManager::initialize()
{
 
  const double timeA = NaluEnv::self().nalu_time();

  // memory diagnostic
  if ( realm_.get_activate_memory_diagnostic() ) {
    NaluEnv::self().naluOutputP0() << "NaluMemory::NonConformalManager::initialize() Begin: " << std::endl;
    realm_.provide_memory_summary();
  }

  elemsToGhost_.clear();

  // loop over nonConformalInfo and initialize to update the elemsToGhost_ vector.
  for ( size_t k = 0; k < nonConformalInfoVec_.size(); ++k )
    nonConformalInfoVec_[k]->initialize();
 
  std::vector<stk::mesh::EntityKey> recvGhostsToRemove;

  if (nonConformalGhosting_ != NULL) {
    stk::mesh::EntityProcVec currentSendGhosts;

    nonConformalGhosting_->send_list(currentSendGhosts);
  
    // We want both elemsToGhost_ to only contain elements not already ghosted, and
    // a list of receive-ghosts that no longer need to be ghosted.
    compute_precise_ghosting_lists(realm_.bulk_data(), elemsToGhost_,
                                   currentSendGhosts, recvGhostsToRemove);
  }

  // check for ghosting need
  size_t local[2] = {elemsToGhost_.size(), recvGhostsToRemove.size()};
  size_t global[2] = {0, 0};
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), local, global, 2);

  if (global[0] > 0 || global[1] > 0) {
      NaluEnv::self().naluOutputP0() << "NonConformal alg will ghost a new number of entities: "
                    << global[0]<< " and remove "<<global[1]<< " entities from ghosting."  << std::endl;

      manage_ghosting(recvGhostsToRemove);
  }
  else {
    NaluEnv::self().naluOutputP0() << "NonConformal alg will NOT ghost entities. " << std::endl;
  }

  // ensure that the coordinates for the ghosted elements (required for the fine search) are up-to-date
  if (nonConformalGhosting_ != NULL) {
    VectorFieldType *coordinates 
      = realm_.bulk_data().mesh_meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
    std::vector<const stk::mesh::FieldBase*> fieldVec = {coordinates};
    stk::mesh::communicate_field_data(*nonConformalGhosting_, fieldVec);
  }

  // complete search
  for ( size_t k = 0; k < nonConformalInfoVec_.size(); ++k )
    nonConformalInfoVec_[k]->complete_search();

  // check for reuse
  bool canReuse = true;
  for ( size_t k = 0; k < nonConformalInfoVec_.size(); ++k )
    canReuse &=nonConformalInfoVec_[k]->canReuse_;
  
  // reset all reusage flags if all are not true
  if ( !canReuse ) {
    for ( size_t k = 0; k < nonConformalInfoVec_.size(); ++k )
      nonConformalInfoVec_[k]->canReuse_ = false;
  }
  
  // Provide diagnosis
  if ( ncAlgDetailedOutput_ ) {
    for ( size_t k = 0; k < nonConformalInfoVec_.size(); ++k )
      nonConformalInfoVec_[k]->provide_diagnosis();
  }

  // error check for coincident nodes
  if ( ncAlgCoincidentNodesErrorCheck_ ) {
    size_t l_problemNodes = 0; size_t g_problemNodes = 0;
    for ( size_t k = 0; k < nonConformalInfoVec_.size(); ++k )
      l_problemNodes += nonConformalInfoVec_[k]->error_check();
    
    // report and terminate if there is an issue
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
    stk::all_reduce_sum(comm, &l_problemNodes, &g_problemNodes, 1);
    if ( g_problemNodes > 0 ) {
      NaluEnv::self().naluOutputP0() << "NonConformalManager::Error() Too many coincident nodes found on NCAlg interface(s): " 
                                     << g_problemNodes <<  " ...ABORTING..." << std::endl;
      throw std::runtime_error("NonConformalManager::Error() Please remesh taking care to avoid coincident nodes");
    }
  }

  // memory diagnostic
  if ( realm_.get_activate_memory_diagnostic() ) {
    NaluEnv::self().naluOutputP0() << "NaluMemory::NonConformalManager::initialize() End: " << std::endl;
    realm_.provide_memory_summary();
  }
  
  // end time
  const double timeB = NaluEnv::self().nalu_time();
  realm_.timerNonconformal_ += (timeB-timeA);
}

//--------------------------------------------------------------------------
//-------- manage_ghosting -------------------------------------------------
//--------------------------------------------------------------------------
void
NonConformalManager::manage_ghosting(std::vector<stk::mesh::EntityKey>& recvGhostsToRemove)
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  bulk_data.modification_begin();

  if ( nonConformalGhosting_ == NULL) {
    // create new ghosting
    std::string theGhostName = "nalu_nonConformal_ghosting";
    nonConformalGhosting_ = &bulk_data.create_ghosting( theGhostName );
  }
    
  bulk_data.change_ghosting( *nonConformalGhosting_, elemsToGhost_, recvGhostsToRemove);
  
  bulk_data.modification_end();

  populate_ghost_comm_procs(bulk_data, *nonConformalGhosting_, ghostCommProcs_);
}

} // namespace nalu
} // namespace sierra
