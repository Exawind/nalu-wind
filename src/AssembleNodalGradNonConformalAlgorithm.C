// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <AssembleNodalGradNonConformalAlgorithm.h>
#include <Algorithm.h>
#include <DgInfo.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradNonConformalAlgorithm - assemble bc nodal grad; both
//                                          edge and elem
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradNonConformalAlgorithm::AssembleNodalGradNonConformalAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  ScalarFieldType* scalarQ,
  VectorFieldType* dqdx)
  : Algorithm(realm, part),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    dualNodalVolume_(NULL),
    exposedAreaVec_(NULL)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(
    meta_data.side_rank(), "exposed_area_vector");

  // what do we need ghosted for this alg to work?
  ghostFieldVec_.push_back(scalarQ_);
  ghostFieldVec_.push_back(dualNodalVolume_);
  ghostFieldVec_.push_back(dqdx_);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradNonConformalAlgorithm::
  ~AssembleNodalGradNonConformalAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradNonConformalAlgorithm::execute()
{

  stk::mesh::BulkData& bulk_data = realm_.bulk_data();
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // ip values; both boundary and opposing surface
  std::vector<double> currentIsoParCoords(nDim);
  std::vector<double> opposingIsoParCoords(nDim);

  // interpolate nodal values to point-in-elem
  const int sizeOfScalarField = 1;

  // nodal fields to gather
  std::vector<double> ws_c_scalarQ;
  std::vector<double> ws_o_scalarQ;

  // parallel communicate ghosted entities
  if (NULL != realm_.nonConformalManager_->nonConformalGhosting_)
    stk::mesh::communicate_field_data(
      *(realm_.nonConformalManager_->nonConformalGhosting_), ghostFieldVec_);

  // iterate nonConformalManager's dgInfoVec
  std::vector<NonConformalInfo*>::iterator ii;
  for (ii = realm_.nonConformalManager_->nonConformalInfoVec_.begin();
       ii != realm_.nonConformalManager_->nonConformalInfoVec_.end(); ++ii) {

    // extract vector of DgInfo
    std::vector<std::vector<DgInfo*>>& dgInfoVec = (*ii)->dgInfoVec_;

    std::vector<std::vector<DgInfo*>>::iterator idg;
    for (idg = dgInfoVec.begin(); idg != dgInfoVec.end(); ++idg) {

      std::vector<DgInfo*>& faceDgInfoVec = (*idg);

      // now loop over all the DgInfo objects on this particular exposed face
      for (size_t k = 0; k < faceDgInfoVec.size(); ++k) {

        DgInfo* dgInfo = faceDgInfoVec[k];

        // extract current/opposing face/element
        stk::mesh::Entity currentFace = dgInfo->currentFace_;
        stk::mesh::Entity opposingFace = dgInfo->opposingFace_;

        // master element
        MasterElement* meFCCurrent = dgInfo->meFCCurrent_;
        MasterElement* meFCOpposing = dgInfo->meFCOpposing_;

        // local ip, ordinals, etc
        const int currentGaussPointId = dgInfo->currentGaussPointId_;
        currentIsoParCoords = dgInfo->currentIsoParCoords_;
        opposingIsoParCoords = dgInfo->opposingIsoParCoords_;

        // mapping from ip to nodes for this ordinal
        const int* faceIpNodeMap = meFCCurrent->ipNodeMap();

        // extract some master element info
        const int currentNodesPerFace = meFCCurrent->nodesPerElement_;
        const int opposingNodesPerFace = meFCOpposing->nodesPerElement_;

        // algorithm related; face
        ws_c_scalarQ.resize(currentNodesPerFace);
        ws_o_scalarQ.resize(opposingNodesPerFace);

        double* p_c_scalarQ = &ws_c_scalarQ[0];
        double* p_o_scalarQ = &ws_o_scalarQ[0];

        // gather current face data
        stk::mesh::Entity const* current_face_node_rels =
          bulk_data.begin_nodes(currentFace);
        const int current_num_face_nodes = bulk_data.num_nodes(currentFace);
        for (int ni = 0; ni < current_num_face_nodes; ++ni) {
          stk::mesh::Entity node = current_face_node_rels[ni];
          // gather...
          p_c_scalarQ[ni] = *stk::mesh::field_data(*scalarQ_, node);
        }

        // gather opposing face data
        stk::mesh::Entity const* opposing_face_node_rels =
          bulk_data.begin_nodes(opposingFace);
        const int opposing_num_face_nodes = bulk_data.num_nodes(opposingFace);
        for (int ni = 0; ni < opposing_num_face_nodes; ++ni) {
          stk::mesh::Entity node = opposing_face_node_rels[ni];
          // gather...
          p_o_scalarQ[ni] = *stk::mesh::field_data(*scalarQ_, node);
        }

        double currentScalarQBip = 0.0;
        meFCCurrent->interpolatePoint(
          sizeOfScalarField, &(dgInfo->currentIsoParCoords_[0]),
          &ws_c_scalarQ[0], &currentScalarQBip);

        double opposingScalarQBip = 0.0;
        meFCOpposing->interpolatePoint(
          sizeOfScalarField, &(dgInfo->opposingIsoParCoords_[0]),
          &ws_o_scalarQ[0], &opposingScalarQBip);

        const double ncScalarQ = 0.5 * (currentScalarQBip + opposingScalarQBip);

        // extract pointers to nearest node fields
        const int nn = faceIpNodeMap[currentGaussPointId];
        stk::mesh::Entity nNode = current_face_node_rels[nn];
        const double volNN = *stk::mesh::field_data(*dualNodalVolume_, nNode);
        double* dqdx = stk::mesh::field_data(*dqdx_, nNode);

        // nearest node inverse volume
        double inv_volNN = 1.0 / volNN;

        // pointer to face data
        const double* c_areaVec =
          stk::mesh::field_data(*exposedAreaVec_, currentFace);

        // assemble to nearest node
        for (int j = 0; j < nDim; ++j) {
          dqdx[j] +=
            ncScalarQ * c_areaVec[currentGaussPointId * nDim + j] * inv_volNN;
        }
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
